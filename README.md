# Local LLM speed on an integrated GPU — 23 models, one mini PC

Generation speed for 23 local models on a single AMD mini PC with an **integrated**
Radeon 890M. No discrete graphics card, 3–8 W at idle, about 900 EUR of hardware.

I run this machine at home as a hobby. I measured the models one by one in
**September 2026**, wrote the numbers down, and this repository is that sheet
plus the script you need to repeat the method on your own box.

- **[`data/local-llm-speed-ryzen-ai-9-hx-470-september-2026.csv`](data/local-llm-speed-ryzen-ai-9-hx-470-september-2026.csv)** — the 23 rows, machine readable
- **[`scripts/measure_tps.py`](scripts/measure_tps.py)** — the measurement script, standard library only
- **[Known gaps](#known-gaps)** — read this before you cite anything here

## The machine

| | |
|---|---|
| CPU | AMD Ryzen AI 9 HX 470, 24 cores |
| GPU | integrated Radeon 890M. No discrete card |
| RAM | 32 GB shared with the CPU, at the time of these measurements |
| Storage | 1 TB NVMe |
| OS | Ubuntu Server, about 400 MB of RAM at idle |
| Runtime | llama.cpp, router mode: one model resident, 30–60 s cold start |
| Power | 3–8 W idle, about 130 W at full load |
| Price | about 900 EUR in Europe |

The iGPU can address about **19 GB** of that shared memory. That number is a
firmware and driver limit, not a share of what is installed: I later moved the
same machine to **64 GB** (two 32 GB SODIMMs) and the aperture did not change,
while generation speed stayed the same within 1%. More RAM buys you context and
instant model switching, not tokens per second.

## The table

Fastest first. Generation speed only — see [Known gaps](#known-gaps).

| Model | Total / active params | Size on disk | tok/s | Architecture |
|---|---|---|---|---|
| granite-4.2-8b | 8B dense | ~4 GB | **45.7** | dense, ngl=99 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | 30B / 3B | 18 GB | 29.9–32.9 (best 31.9) | MoE, MTP n2 |
| Qwen3-Coder-30B-A3B | 30B / 3B | 18 GB | 30.4 | MoE |
| gpt-oss-20b | 20B / not recorded | 11 GB | 30.4 | MoE, ngl=99 |
| Qwen3.6-35B-A3B | 35B / 3B | 21 GB | 30.4 base, 27.4 MTP | MoE |
| Qwen3.8-9B | 9B dense | 5.4 GB | 27.7–28.1 | dense, DFlash |
| gemma-4-26B-A4B-it-qat | 26B / 4B | 14 GB | ~28 | MoE, QAT |
| Moonlight-16B-A3B (APEX-i-quality) | 16B / 3B | 11.9 GB | 26.57 | MoE, MLA |
| SmallThinker-21B-A3B (QAT) | 21B / 3B | 12.2 GB | 26.23 | MoE, QAT |
| Nail-35B-A3B (MTP) | 35B / 3B | 22.8 GB | 24.17 | MoE, MTP |
| Tiel-Coder-35B-A3B (resident) | 35B / 3B | 21.7 GB | 23.11 | MoE, MTP |
| ornith-1.5-35b | 35B / 3B | 21 GB | 22.6 | MoE |
| Nex-N2.5-mini | not recorded / 3B | 21.2 GB | 21.11 | MoE |
| Nail-35B-A3B (base) | 35B / 3B | 22.4 GB | 17.86 | MoE |
| Qwen-AgentWorld-35B-A3B | 35B / 3B | 21 GB | 16.3 | MoE |
| ornith-1.5-9b-mtp | 9B dense | 5.3 GB | ~14 long context, 29 warm short | dense |
| THUDM GLM-Z1-9B | 9B dense | 5.8 GB | 13.4 long context, 15.7 warm | dense, CPU |
| gemma-4-12b-it | 12B dense | 7.9 GB | 11.3 long context, up to 34 warm short | dense, CPU |
| Qwen3.8-27B-MTP | 27B dense | 16 GB | 7.6 | dense |
| gemma-4-31B-it | 31B dense | 18 GB | 6.1–6.4 (DSpark) | dense |
| Muse-Glimmer-30B-UD | 30B dense | 15 GB | 4.2 | dense |
| DeepSeek-R1-Distill-Qwen-32B | 32B dense | 19 GB | ~3–4 | dense |
| Qwen3.6-27B-Fable | 27B dense | 17 GB | ~3.5 | dense |

Quantisation is in the CSV, not in this table, because I only know it for some
rows: the six Qwen rows are **Q4_K_M**, `gpt-oss-20b` is native **MXFP4**, and
`gemma-4-26B-A4B-it-qat` and `SmallThinker-21B-A3B` are **trained** at 4 bits
rather than compressed to it. Every other row says `not recorded`, and that
means exactly what it says.

## What the table shows

**1. Architecture decides the speed, not the file size.** The clean proof is
inside one family. All six Qwen rows sit within **0.3 bits per weight** of each
other — 4.7 to 5.1 reading GB as 10⁹ bytes, 5.1 to 5.4 reading the sizes as GiB,
the same narrow band either way — and their speeds run from 30.4 down to ~3.5.
Same family, same quantisation, an order of magnitude apart. `Qwen3-Coder-30B-A3B`
is a *bigger* file than dense `Qwen3.8-27B` and runs **four times faster**,
because it activates 3B parameters per token instead of 27B. (Bits per weight is
arithmetic on the two published columns, size ÷ parameters; it is not a separate
measurement.)

**2. Eleven of the thirteen mixture-of-experts rows land at 21–33 tok/s.** Two
fall below: `Nail-35B-A3B (base)` at 17.86 and `Qwen-AgentWorld-35B-A3B` at
16.3. Both are 35B names, so size is not irrelevant — it just stops being the
first-order term. And note the class carefully: only eleven of those thirteen
rows are *~3B-active*. `gpt-oss-20b` has no active-parameter count in my notes
and `gemma-4-26B-A4B-it-qat` is 4B active, so "~3B-active MoE lands at 21–33"
would be the wrong sentence for the right data.

**3. Dense models above 9B are memory-bandwidth bound — in long context.** In
long-context runs every dense row above 9B is at **11.3 tok/s or below**: 11.3,
7.6, 6.1–6.4, 4.2, ~3–4, ~3.5. The qualifier matters, because the same
`gemma-4-12b-it` reaches **up to 34 tok/s on a warm short prompt**. Any version
of this rule without "long context" is refuted by its own table.

**4. A dense 30B at reduced precision is 15–19 GB on disk here**, not the 18–22 GB
that gets repeated. `Muse-Glimmer-30B-UD` is 30B in **15 GB** — and still only
4.2 tok/s, which is the point of finding 1.

**5. The fastest model on this box is `granite-4.2-8b` at 45.7 tok/s.** Not any
Qwen, and not `gpt-oss-20b`.

My own reading threshold: above **10 tok/s** a local model already feels natural
to read along with. Eighteen of these 23 rows clear it on an integrated GPU; the
five that do not are all dense and all above 26B.

## Power, and the arithmetic people ask for

Measured and published: **3–8 W idle, about 130 W at full load.** Everything
below is arithmetic on those two numbers at 0.30 EUR/kWh, shown so you can check
it. **Nothing here was metered at the socket.**

| | |
|---|---|
| Switched on all year, generating nothing | 8 W × 8760 h = 70 kWh ≈ **21 EUR/year** |
| Same, plus one hour of generation a day at 130 W | ≈ **34 EUR/year** |
| Energy per million generated tokens, 30B-A3B at 30.4 tok/s | ≈ **0.36 EUR** |
| Same, dense 9B at 27.7–28.1 tok/s | ≈ **0.39 EUR** |
| Same, dense 27B at 7.6 tok/s | ≈ **1.43 EUR** |

The per-token rows use the 130 W peak as an upper bound, so treat them as a
ceiling. The last three lines are the same machine and the same tariff, four
times apart — which is finding 1 again, in euros.

## Repeating the method

`scripts/measure_tps.py` measures single-stream generation speed against any
OpenAI-compatible endpoint: llama.cpp `llama-server`, LiteLLM, vLLM, Ollama's
`/v1`. Python 3.9+, standard library only, nothing to install.

```bash
# what does the endpoint serve?
./scripts/measure_tps.py --base-url http://localhost:8080/v1 --list-models

# one model, one warmup run, three measured runs
./scripts/measure_tps.py --base-url http://localhost:8080/v1 \
    --model qwen3.8-9b --repeat 3 --max-tokens 256

# every model it lists, long prompt, appended to a CSV
./scripts/measure_tps.py --base-url http://localhost:8080/v1 --all-models \
    --prompt-file long_context.txt --csv my-results.csv
```

It reports three numbers per run:

- **tok/s** — generated tokens over the first-token-to-last-token interval. Prompt
  processing is deliberately outside this window, because that is what the table above means.
- **ttft_s** — request to first token. This is the part tok/s excludes, and in an
  agentic loop with a long prompt it is most of your wait.
- **total_s** — wall clock for the whole request.

It prefers the server's own `completion_tokens` and falls back to counting
streamed deltas, printing which one it used. It does not guess anything else.

## Known gaps

The whole value of this sheet is that every line is checkable, so here is what
it does *not* contain.

- **No prompt-processing figure.** There is no `pp512` anywhere in my notes, and
  the tok/s column is not a `llama-bench tg128` row. If you want the standard
  pair, run `llama-bench -m model.gguf -p 512 -n 128 -ngl 99` yourself.
- **The backend is not recorded** — Vulkan or ROCm/HIP. It matters, and I did not
  write it down.
- **The llama.cpp build and commit are not recorded.**
- **Model names are as they appear on my machine.** The upstream Hugging Face id
  is not recorded for any row, and neither is the LiteLLM alias.
- **Quantisation is known for nine of 23 rows** and `not recorded` for the rest.
- **Single stream, one request at a time.** No batching, no concurrency, so none
  of this tells you about throughput under load.
- **Nothing was metered at the socket.** The watt figures are the machine's
  published idle and full-load numbers; every euro in this file is arithmetic.
- **Not a model quality comparison.** Speed only. A fast model that is wrong is
  not useful.
- **One machine, one run, September 2026.** Treat it as one data point.

## Where these numbers were published

The 23-row table was first posted publicly on **13 September 2026**, as an author
comment under my own article:

- **The Cheapest Hardware Setup for Local AI** —
  <https://automationbro.substack.com/p/the-cheapest-hardware-setup-for-local>

The article body carries an **earlier, August** set of measurements and has not
been rewritten, so it is lower in a few places (Qwen3-Coder 21.4 against 30.4
here, dense 9B "about 15 to 20" against 27.7–28.1). The September figures in this
repository are the ones I stand behind. I also write about local AI, automation
and the home stack in Polish at <https://automationbro.pl>.

## Contributing a machine

Different hardware is the most useful thing you can add. Run the script, open an
issue with the **Add a machine** template, or send a pull request with a new CSV
in `data/` named after your box. One file per machine, the same columns, and
`not recorded` wherever you do not have the number. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Scripts under `scripts/`: MIT. Data under `data/` and the text of this README:
CC BY 4.0 — use the table anywhere, with a link back.
