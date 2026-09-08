"""Synchronize the one-cell quality patch into the free-GPU server notebook."""

from pathlib import Path

import jupytext


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "notebooks" / "patches" / "quality_upgrade.py"
SERVER_PATH = ROOT / "notebooks" / "SatQuery_Qwen3VL_Free_GPU_Server.py"
LIVE_PATH = ROOT / "notebooks" / "SatQuery_Live_Quality_V4.ipynb"
SEGMENTATION_SOURCE = ROOT / "notebooks" / "SatQuery_SegFormer_LoveDA_Training.py"
SEGMENTATION_NOTEBOOK = SEGMENTATION_SOURCE.with_suffix(".ipynb")

START = "# %% [markdown]\n# ## 6b. Detailed reports and candidate pixel masks"
END = "# %% [markdown]\n# ## 6c. Learned intent planning"


def main() -> None:
    patch = PATCH_PATH.read_text(encoding="utf-8").rstrip() + "\n"
    source = SERVER_PATH.read_text(encoding="utf-8")
    if START not in source or END not in source:
        raise RuntimeError("Could not locate the bounded quality section in the server source")
    before, remainder = source.split(START, 1)
    _, after = remainder.split(END, 1)
    section = (
        START
        + "\n#\n# Whole-scene semantic masks for supported land-cover classes; Qwen/SAM fallback.\n"
        + "# Run this after section 6. Do not rerun section 6 afterwards without rerunning 6b.\n\n"
        + "# %%\n"
        + patch
        + "\n"
    )
    synchronized = before + section + END + after
    SERVER_PATH.write_text(synchronized, encoding="utf-8", newline="\n")
    notebook = jupytext.reads(
        synchronized.replace("# ruff: noqa: E402\n", "", 1), fmt="py:percent"
    )
    SERVER_PATH.with_suffix(".ipynb").write_text(
        jupytext.writes(notebook, fmt="ipynb"), encoding="utf-8", newline="\n"
    )

    live_source = (
        "# %% [markdown]\n"
        "# # SatQuery live quality v4 upgrade\n"
        "# Run the next cell in an existing Kaggle server session after section 6.\n"
        "# Stop active requests first. It adds a lightweight semantic segmentation model,\n"
        "# keeps Qwen/SAM as fallback, changes no trained adapter weights and creates no endpoint.\n\n"
        "# %%\n"
        + patch
    )
    live_notebook = jupytext.reads(live_source, fmt="py:percent")
    LIVE_PATH.write_text(
        jupytext.writes(live_notebook, fmt="ipynb"), encoding="utf-8", newline="\n"
    )
    segmentation_source = SEGMENTATION_SOURCE.read_text(encoding="utf-8").replace(
        "# ruff: noqa: E402\n", "", 1
    )
    segmentation_notebook = jupytext.reads(segmentation_source, fmt="py:percent")
    segmentation_notebook.cells = [
        cell
        for cell in segmentation_notebook.cells
        if cell.cell_type != "code" or cell.source.strip()
    ]
    SEGMENTATION_NOTEBOOK.write_text(
        jupytext.writes(segmentation_notebook, fmt="ipynb"),
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS: synchronized quality v4, live patch and SegFormer training notebooks"
    )


if __name__ == "__main__":
    main()
