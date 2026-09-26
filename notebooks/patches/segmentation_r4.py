"""R4 helpers embedded after the R3 Trainer definition; no serving/gate changes."""

import copy
import math
import time
from transformers import TrainerCallback


PROTECTED_REFERENCE = None


def protected_checkpoint_score(metrics, reference):
    """Prefer any non-regressing candidate over a better aggregate with regressions.

    This is a selection score, NOT accuracy or calibrated confidence. The original
    release_balance and final release gate stay unchanged.
    """
    names = ("mean_iou", "iou_water", "iou_forest", "iou_agricultural")
    values = [float(metrics.get(name, float("nan"))) for name in names]
    balance = float(metrics.get("release_balance", 0))
    if not all(math.isfinite(value) for value in values + [balance]):
        return -1.0
    if reference is None:
        return min(1.0, max(0.0, balance))
    nonregressing = all(metrics[name] >= reference["eval_" + name] for name in names)
    return 10.0 + balance if nonregressing else min(1.0, max(0.0, balance))


def teacher_support_mask(probabilities, labels, confidence_threshold):
    """Ground truth vetoes incorrect teacher pixels; nodata never contributes."""
    confidence, predicted = probabilities.max(dim=1)
    return (labels != 255) & (predicted == labels) & (confidence >= confidence_threshold)


if not warm_start or warm_start["weights_sha256"] != CFG.required_incumbent_sha256:
    raise RuntimeError("R4 requires the verified R2 incumbent; no random-head training or old optimizer resume.")

# A small frozen copy of the incumbent; no second large VLM and no gradients.
# This reference is NOT registered as a student submodule or saved in its weights.
teacher_model = copy.deepcopy(model).eval().requires_grad_(False)


class R4DiceCETrainer(DiceCETrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        pixels = inputs["pixel_values"]
        loss, outputs = super().compute_loss(
            model, dict(inputs), return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        # Validation loss remains supervised-only and comparable to the incumbent.
        if model.training and CFG.teacher_weight > 0:
            teacher_model.to(pixels.device)
            teacher_model.eval()
            with torch.no_grad():
                teacher_logits = teacher_model(pixel_values=pixels).logits
                teacher_logits = F.interpolate(
                    teacher_logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                if not torch.isfinite(teacher_logits).all():
                    raise FloatingPointError("Non-finite teacher logits; stop rather than corrupt the student.")
                supported = teacher_support_mask(
                    teacher_logits.softmax(dim=1), labels, CFG.teacher_confidence
                )
                temperature = CFG.teacher_temperature
                target = (teacher_logits / temperature).softmax(dim=1)
            if supported.any():
                student = F.interpolate(
                    outputs.logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                divergence = F.kl_div(
                    (student / temperature).log_softmax(dim=1), target, reduction="none"
                ).sum(dim=1)
                loss = loss + CFG.teacher_weight * temperature**2 * divergence[supported].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite R4 loss; preserve diagnostics and stop.")
        return (loss, outputs) if return_outputs else loss


class TrainingTimeBudget(TrainerCallback):
    """Leave time to evaluate/export; never auto-renew a cloud session."""
    def on_train_begin(self, args, state, control, **kwargs):
        self.deadline = time.monotonic() + CFG.training_budget_hours * 3600

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
            print("Training budget reached: evaluate, preserve this checkpoint, then export the selected model.")
        return control


def r4_gradient_preflight(trainer):
    """One training-labelled batch, no optimizer step, restore buffers and RNG."""
    from transformers import set_seed
    saved_buffers = {name: value.detach().clone() for name, value in trainer.model.named_buffers()}
    was_training = trainer.model.training
    try:
        trainer.model.train()
        example = train_dataset[0]
        inputs = {key: value.unsqueeze(0).to(trainer.args.device) for key, value in example.items()}
        trainer.model.zero_grad(set_to_none=True)
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("R4 gradient preflight failed; do not start epochs.")
        if any(p.grad is not None or p.requires_grad for p in teacher_model.parameters()):
            raise RuntimeError("Teacher must remain frozen.")
        print({"R4_gradient_preflight": "PASS", "loss": float(loss.detach()),
               "peak_cuda_allocated_GiB": round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
    finally:
        trainer.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in trainer.model.named_buffers():
                value.copy_(saved_buffers[name])
        trainer.model.train(was_training)
        set_seed(CFG.seed)
