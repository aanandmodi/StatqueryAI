---
title: SatQuery API
emoji: 🌍
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 30m
short_description: Auditable API gateway for SatQuery imagery analysis
pinned: false
license: apache-2.0
---

# SatQuery API

FastAPI-compatible orchestration layer hosted through Gradio Server mode. The Space is explicitly
assigned the second free ZeroGPU slot because unpaid accounts can no longer provision Gradio or
Docker on CPU Basic. Its orchestration routes do not request the GPU; they delegate the three
released VLM tasks to the separate model Space. Uploaded rasters, SQLite job state, and reports are
ephemeral, so clients must keep their source files and download artifacts they need to retain.

Required Space variables and secrets are set by `scripts/deploy_free_backend_space.py`.
