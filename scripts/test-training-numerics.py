"""Small CPU unit tests of notebook definitions. No datasets, downloads, or training run."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def definitions(file, names, **extra):
    tree = ast.parse((ROOT / "notebooks" / file).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(torch=torch, F=F, math=math, np=np, Dataset=Dataset,
                     ResNet18_Weights=ResNet18_Weights, resnet18=resnet18, **extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, "exec"), namespace)
    return namespace


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(1, 16, 1)

    def forward(self, inputs):
        value = sum(item.mean(1, keepdim=True) for item in inputs.values()) / len(inputs)
        return [F.adaptive_avg_pool2d(self.projection(value), (4, 4)).flatten(2).transpose(1, 2)]


class TrainingNumericsTests(unittest.TestCase):
    def test_fusion_versions_all_modalities_and_strict_reload(self):
        ns = definitions("SatQuery_TerraMind_Fusion_Training.py", {"FusionExpert", "segmentation_loss"})
        for version in ("legacy", "conv-r2"):
            model = ns["FusionExpert"](TinyBackbone(), 16, 2, 32, decoder_version=version)
            for mode in ("fused", "s2", "s1"):
                model.zero_grad(set_to_none=True)
                kwargs = {"s2": torch.randn(2, 13, 32, 32) if mode != "s1" else None,
                          "s1": torch.randn(2, 2, 32, 32) if mode != "s2" else None}
                output = model(**kwargs)
                self.assertEqual(tuple(output.shape), (2, 2, 32, 32))
                loss = ns["segmentation_loss"](output, torch.randint(-1, 2, (2, 32, 32)))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            reloaded = ns["FusionExpert"](TinyBackbone(), 16, 2, 32, decoder_version=version)
            reloaded.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(model(**kwargs), reloaded(**kwargs))

    def test_fusion_loss_all_ignore_and_half_precision_large_reductions(self):
        loss_fn = definitions("SatQuery_TerraMind_Fusion_Training.py", {"segmentation_loss"})["segmentation_loss"]
        logits = torch.zeros(4, 2, 256, 256, dtype=torch.float16, requires_grad=True)
        loss = loss_fn(logits, torch.ones(4, 256, 256, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())
        logits.grad = None
        ignored = loss_fn(logits, torch.full((4, 256, 256), -1, dtype=torch.long))
        self.assertEqual(float(ignored.detach()), 0.)
        ignored.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(FloatingPointError):
            loss_fn(torch.full((1, 2, 4, 4), float("nan")), torch.zeros(1, 4, 4, dtype=torch.long))

    def test_change_warm_start_new_decoder_cache_and_gradient(self):
        ns = definitions("SatQuery_ChangeVQA_Training.py", {"ChangeExpert", "objective"}, CFG=SimpleNamespace(mask_weight=2.0))
        old = ns["ChangeExpert"](10, 4, pretrained=False).eval()
        model = ns["ChangeExpert"](10, 4, pretrained=False, decoder_version="multiscale-r2").eval()
        result = model.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(key.startswith(("laterals.", "detail_head.")) for key in result.missing_keys))
        a, b, tokens = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), torch.ones(2, 4, dtype=torch.long)
        answers, masks = model(a, b, tokens)
        self.assertEqual(tuple(masks.shape), (2, 1, 64, 64))
        visual, cached_mask = model.encode_pair(a, b)
        torch.testing.assert_close(answers, model.answer_from_visual(visual, tokens))
        torch.testing.assert_close(masks, cached_mask)
        loss = ns["objective"](answers, masks, torch.tensor([0, 1]), torch.ones_like(masks))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.laterals.parameters()))
        copy = ns["ChangeExpert"](10, 4, pretrained=False, decoder_version="multiscale-r2").eval()
        copy.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(copy(a, b, tokens)[1], masks)

    def test_flood_dataset_sanitizes_raw_nan_before_interpolation(self):
        s2, s1, labels = np.ones((13, 4, 4), np.float32), np.ones((2, 4, 4), np.float32), np.ones((4, 4), np.int64)
        s2[0, 1, 1], s1[0, 2, 2] = np.nan, np.inf

        class Source:
            def __init__(self, value):
                self.value, self.count = value, value.shape[0] if value.ndim == 3 else 1
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, index=None): return self.value.copy()
            def read_masks(self): return np.full_like(self.value, 255, dtype=np.uint8)
            def dataset_mask(self): return np.full((4, 4), 255, np.uint8)

        ns = definitions("SatQuery_TerraMind_Fusion_Training.py", {"FloodDataset"},
            CFG=SimpleNamespace(image_size=4), triplet=lambda _: ("s2", "s1", "label"), same_grid=lambda _: None,
            rasterio=SimpleNamespace(open=lambda path: Source({"s2": s2, "s1": s1, "label": labels}[path])),
            TM_S2_MEAN=torch.ones(13), TM_S2_STD=torch.ones(13), TM_S1_MEAN=torch.ones(2), TM_S1_STD=torch.ones(2))
        sample = ns["FloodDataset"](["fixture"], False)[0]
        self.assertTrue(torch.isfinite(sample["s2"]).all() and torch.isfinite(sample["s1"]).all())
        self.assertEqual(int(sample["target"][1, 1]), -1)
        self.assertEqual(int(sample["target"][2, 2]), -1)
        self.assertEqual(int((sample["target"] == 1).sum()), 14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
