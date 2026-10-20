# Contributing a machine

One machine, one CSV file. That is the whole model.

## The short way

Open an issue with the **Add a machine** template and paste your numbers. I will
add the file.

## The pull request way

1. Run the script. A warmup run plus three measured runs per model is enough:

   ```bash
   ./scripts/measure_tps.py --base-url http://localhost:8080/v1 --all-models \
       --repeat 3 --warmup 1 --csv data/local-llm-speed-<your-machine>.csv
   ```

   The script writes its own per-run columns. Convert them to the table schema,
   or paste the raw file and say so — I would rather have raw numbers than tidy
   wrong ones.

2. Name the file `data/local-llm-speed-<machine>-<month-year>.csv`.

3. Use the columns in [`data/README.md`](data/README.md). Write `not recorded`
   wherever you do not have the number. Do not copy a value down from the row
   above, and do not fill a column by inference.

4. Add one row to the file table in `data/README.md` with the machine, the date
   and the row count.

5. In the pull request, say what you ran: the runtime and its version, the
   backend (Vulkan, ROCm, CUDA, Metal, CPU), the quantisation, and whether the
   prompt was short or long. Those four decide whether your numbers can be
   compared to anyone else's.

## What gets rejected

- Numbers you did not measure yourself.
- A tok/s figure with no stated prompt length or context regime. The same model
  can differ threefold between a warm short prompt and a long context.
- Estimates presented as measurements. An estimate is welcome if it is labelled
  as one.
- Quality or benchmark scores. This repository is about speed on your own
  hardware, nothing else.

## Corrections

If a number here is wrong, open an issue and say which row and why. A correction
to my own rows is as welcome as a new machine — several of the rules in the
README were rewritten after the table refuted them.
