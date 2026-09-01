---
title: SatQuery API
emoji: 🌍
colorFrom: slate
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# SatQuery API

FastAPI orchestration layer for SatQuery. This Space runs on free CPU hardware and delegates the
three released VLM tasks to the separate ZeroGPU Space. Uploaded rasters, SQLite job state, and
reports are ephemeral on the free Space filesystem; clients must keep their source files and
download reports they need to retain.

Required Space variables and secrets are set by `scripts/deploy_free_backend_space.py`.

