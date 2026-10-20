#!/usr/bin/env python3
"""Measure single-stream generation speed against an OpenAI-compatible endpoint.

Works with llama.cpp `llama-server`, LiteLLM, vLLM, Ollama (`/v1`) and anything
else that speaks `POST /v1/chat/completions` with `stream: true`.

What it reports, per model:

  tok/s        (generated tokens - 1) / (time of last token - time of first token)
               The first token is the start of the clock, so it cannot be inside
               the interval it measures. On 256 tokens this is a 0.4% difference.
  ttft_s       seconds from request to the first token (prompt processing + queue)
  total_s      wall clock for the whole request

tok/s deliberately excludes prompt processing, because that is how the numbers
in `data/` were taken. ttft_s is reported next to it so you can see the part
that is excluded. On an agentic loop with a long prompt, ttft_s is most of
your wait, not tok/s.

Standard library only. No pip install.

Examples
--------
  # list what the endpoint serves
  ./measure_tps.py --base-url http://localhost:8080/v1 --list-models

  # one model, three repeats, 256 generated tokens
  ./measure_tps.py --base-url http://localhost:8080/v1 \
      --model qwen3.8-9b --repeat 3 --max-tokens 256

  # every model the endpoint lists, long prompt, append to a CSV
  ./measure_tps.py --base-url http://localhost:8080/v1 --all-models \
      --prompt-file long_context.txt --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Explain, in about 200 words, why a mixture-of-experts model generates "
    "faster than a dense model of the same size on a memory-bandwidth-bound "
    "machine. Do not use bullet points."
)

CSV_FIELDS = [
    "timestamp",
    "model",
    "run",
    "prompt_chars",
    "max_tokens",
    "completion_tokens",
    "token_count_source",
    "ttft_s",
    "generation_s",
    "total_s",
    "tps",
]


def http_json(url: str, api_key: str | None, payload: dict | None = None, timeout: int = 600):
    """GET when payload is None, POST otherwise. Returns parsed JSON."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def list_models(base_url: str, api_key: str | None) -> list[str]:
    body = http_json(f"{base_url}/models", api_key, timeout=60)
    return [m["id"] for m in body.get("data", []) if "id" in m]


def one_run(base_url, api_key, model, prompt, max_tokens, temperature, timeout):
    """Stream one completion and time it. Returns a dict of measurements."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # llama.cpp and vLLM honour this and give a real token count in the
        # final chunk. Servers that ignore it fall back to counting deltas.
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    deltas = 0
    usage_tokens = None
    t_first = None
    t_last = None
    text_parts: list[str] = []

    t_start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens", usage_tokens)
            for choice in obj.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if not piece:
                    continue
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                deltas += 1
                text_parts.append(piece)
    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError(f"{model}: the stream carried no content")

    # Use the server's own token count when it gives one. Counting SSE deltas
    # is a fallback: a delta is usually one token, but no API promises that.
    if usage_tokens:
        tokens, source = usage_tokens, "server usage"
    else:
        tokens, source = deltas, "delta count"

    # Generation time is first token to last token. With one token there is no
    # interval to measure, so tok/s is undefined rather than infinite.
    generation_s = (t_last - t_first) if tokens > 1 else None
    tps = (tokens - 1) / generation_s if generation_s and generation_s > 0 else None

    return {
        "model": model,
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "completion_tokens": tokens,
        "token_count_source": source,
        "ttft_s": round(t_first - t_start, 3),
        "generation_s": round(generation_s, 3) if generation_s else "",
        "total_s": round(t_end - t_start, 3),
        "tps": round(tps, 2) if tps else "",
        "text": "".join(text_parts),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single-stream generation speed for a local LLM endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
                   help="endpoint root, ending in /v1 (default: %(default)s)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="bearer token, if the endpoint wants one. Prefer the OPENAI_API_KEY env var")
    p.add_argument("--model", action="append", dest="models", metavar="NAME",
                   help="model to measure. Repeat the flag for several")
    p.add_argument("--all-models", action="store_true", help="measure every model /v1/models lists")
    p.add_argument("--list-models", action="store_true", help="print the model list and exit")
    p.add_argument("--prompt", default=None, help="prompt text (default: a short built-in prompt)")
    p.add_argument("--prompt-file", default=None, help="read the prompt from this file")
    p.add_argument("--max-tokens", type=int, default=256, help="tokens to generate (default: %(default)s)")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default: %(default)s)")
    p.add_argument("--repeat", type=int, default=3, help="measured runs per model (default: %(default)s)")
    p.add_argument("--warmup", type=int, default=1,
                   help="unmeasured runs first, to load the model and warm the page cache (default: %(default)s)")
    p.add_argument("--timeout", type=int, default=600, help="request timeout in seconds (default: %(default)s)")
    p.add_argument("--csv", default=None, help="append every run to this CSV file")
    p.add_argument("--json", dest="json_out", default=None, help="write all results to this JSON file")
    p.add_argument("--print-output", action="store_true", help="also print what the model generated")
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        if args.list_models or args.all_models:
            available = list_models(base_url, args.api_key)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"cannot reach {base_url}/models: {exc}", file=sys.stderr)
        return 2

    if args.list_models:
        for name in available:
            print(name)
        return 0

    models = list(args.models or [])
    if args.all_models:
        models += [m for m in available if m not in models]
    if not models:
        p.error("give --model NAME, or --all-models, or --list-models")

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()
    else:
        prompt = args.prompt or DEFAULT_PROMPT

    print(f"endpoint   {base_url}")
    print(f"prompt     {len(prompt)} characters")
    print(f"generate   up to {args.max_tokens} tokens, temperature {args.temperature}")
    print(f"runs       {args.warmup} warmup + {args.repeat} measured per model")
    print()

    results = []
    failures = 0
    for model in models:
        runs = []
        try:
            for _ in range(max(args.warmup, 0)):
                one_run(base_url, args.api_key, model, prompt,
                        args.max_tokens, args.temperature, args.timeout)
            for i in range(1, args.repeat + 1):
                row = one_run(base_url, args.api_key, model, prompt,
                              args.max_tokens, args.temperature, args.timeout)
                row["run"] = i
                row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                runs.append(row)
                print(f"  {model}  run {i}  {row['tps'] or 'n/a'} tok/s"
                      f"  ttft {row['ttft_s']} s  {row['completion_tokens']} tokens"
                      f"  ({row['token_count_source']})")
                if args.print_output:
                    print(f"    > {row['text'][:400]}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError) as exc:
            failures += 1
            print(f"  {model}  FAILED: {exc}", file=sys.stderr)

        measured = [r["tps"] for r in runs if isinstance(r["tps"], (int, float))]
        if measured:
            print(f"  {model}  median {statistics.median(measured)} tok/s"
                  f"  (n={len(measured)}, min {min(measured)}, max {max(measured)})")
        print()
        results.extend(runs)

    if args.csv and results:
        new_file = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerows(results)
        print(f"appended {len(results)} rows to {args.csv}")

    if args.json_out and results:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")

    if failures and not results:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
