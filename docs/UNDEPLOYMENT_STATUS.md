# Hosted-resource status

Last checked: 2026-09-02.

| Resource | State | Publicly callable? | Repository action |
|---|---|---:|---|
| `aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora` | Public model artifact at pinned SHA | No compute endpoint | Preserved; required for notebook downloads |
| `aanandmodi/satquery-qwen3vl-space` | Private static Space | No | Archived, not deleted |
| `aanandmodi/StatqueryAI` | Private static Space | No | Archived, not deleted |
| Previous ChatGPT Site | Owner-only | No external viewers | Local hosting binding removed |
| Kaggle/ngrok service | Off until user runs notebook | Temporary only | Notebook cell 10 controls lifetime |

The two Hugging Face Spaces are private because static Spaces cannot be paused. The available Sites
connector provides access control but no unpublish/delete primitive; the existing Site is therefore
owner-only and receives no updates. There is no `.openai/hosting.json`, deployment package, or
deployment helper left in the working repository.
