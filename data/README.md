# Data

One file per measurement run.

| file | machine | date | rows |
|---|---|---|---|
| `local-llm-speed-ryzen-ai-9-hx-470-september-2026.csv` | AMD Ryzen AI 9 HX 470, integrated Radeon 890M, 32 GB shared | September 2026 | 23 |

## Columns

| column | meaning |
|---|---|
| `model` | the name as it appears on my machine, including the build suffix |
| `family` | model family and size, written out |
| `params_total_b` | total parameters, in billions. `not recorded` where I do not have the number |
| `params_active_b` | active parameters per token, in billions. Equal to the total on dense models, because a dense model activates every weight |
| `architecture` | `dense` or `MoE` |
| `quantisation` | only where I know it. `not recorded` is not a hint — it means I did not write it down |
| `size_on_disk_gb` | the GGUF size on disk |
| `tps_reported` | the figure exactly as I recorded it, text and all. This is the source of truth |
| `tps_min`, `tps_max` | the same figure split into numbers so you can sort and plot. Equal to each other when one value was recorded |
| `regime` | what the prompt looked like. A short warm prompt and a long context give very different numbers on the same model |
| `runtime_notes` | flags and build details that affect the figure |
| `intended_use` | what I keep the model for |

## Reading rules

- `tps_reported` wins. `tps_min` / `tps_max` exist only to make the file sortable.
- A `~` in `tps_reported` means the figure is approximate, and `tps_min` / `tps_max` inherit that.
- `not recorded` never means zero, and it never means "the same as the row above".
- Generation speed only. There is no prompt-processing figure in this file. See the main README.
