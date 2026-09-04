"""Mechanically embed the maintained Kaggle quality cell and regenerate the paired notebook."""
from pathlib import Path

import jupytext

root = Path(__file__).resolve().parents[1]
script = root / "notebooks/SatQuery_Qwen3VL_Free_GPU_Server.py"
patch = root / "notebooks/patches/quality_upgrade.py"
source = script.read_text(encoding="utf-8")
start = "# %% [markdown]\n# ## 6b. Detailed reports and candidate pixel masks"
end = "# %% [markdown]\n# ## 7. Verify the local HTTP contract"
if start in source:
    before, rest = source.split(start, 1)
    source = before + end + rest.split(end, 1)[1]
source = source.replace(end,
    start + "\n#\n# No retraining or deployment. SAM refines Qwen proposals, not verified semantic labels.\n"
    "# Run this after section 6. Do not rerun section 6 afterwards without rerunning 6b.\n\n"
    "# %%\n" + patch.read_text(encoding="utf-8") + "\n\n" + end,
    1,
)
script.write_text(source, encoding="utf-8", newline="\n")
# A Python lint directive inside YAML must not become a visible raw notebook cell.
notebook = jupytext.reads(source.replace("# ruff: noqa: E402\n", "", 1), fmt="py:percent")
script.with_suffix(".ipynb").write_text(
    jupytext.writes(notebook, fmt="ipynb"), encoding="utf-8", newline="\n",
)
print("Synced quality cell; notebook code cells:", sum(c.cell_type == "code" for c in notebook.cells))
