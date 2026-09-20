# Pressure-test agent

Adversarial harness for the decode path and the UI. It is not a second test
suite: `tests/` asserts that things work, this asserts that things **fail
loudly instead of quietly returning a wrong number**.

```bash
python -m agents.pressure_test                  # everything
python -m agents.pressure_test --quick          # fast subset (pre-push hook)
python -m agents.pressure_test --probes ingest,ui
python -m agents.pressure_test --fail-on high   # non-zero exit, for cron/CI
```

Reports land in `reports/pressure_test/`: `report.md` to read, `report.json`
to diff between runs.

## Probes

| probe | asks |
|---|---|
| `correctness` | Does every count match a brute-force oracle, on hostile-but-real inputs? |
| `causality`   | Can anything at time *t* see past *t*? Can a fit see the held-out rows? |
| `ingest`      | Does Kilosort/Phy sorted output actually get in? |
| `latency`     | What does one decode step cost, and how does it scale? |
| `ui`          | Does every page render, and does a blocked page say what to do next? |
| `hygiene`     | What is committed that should not be? |

### The oracle

`synth.SpikeSet.truth_counts` counts spikes with a Python loop and shares no
code with the pipeline. Every count the repo produces is compared against it.
That is the whole design: a fast implementation is only trustworthy against a
slow one that is obviously right.

### The adversarial inputs

Each is a condition that occurs in real recordings, not a fuzzer artifact:
unsorted and locally-out-of-order arrival (threaded sorters, ZMQ), duplicate
timestamps, a single far-future sample (clock glitch), float32 sample-clock
rounding, negative pre-trigger times, silent units, sparse non-contiguous Phy
cluster ids, and vendor column naming.

## Severity

| level | meaning |
|---|---|
| CRITICAL | returns a wrong number, or blocks real data, with no error raised |
| HIGH | wrong under a realistic condition, or an unenforced documented contract |
| MEDIUM | degrades, misleads, or costs the user real time |
| LOW | rough edge |
| INFO | measurement recorded so regressions show as a diff |

INFO findings are not filler. `causality` emitting "No causal leakage
detected" and `latency` emitting its scaling table are the records that make
the *next* run meaningful.

## Adding a probe

Subclass `Probe`, implement `checks()` yielding `Finding`s, register it in
`runner._load_probes`. Wrap each check in `self.check(...)` so a broken check
degrades into a LOW finding instead of taking down the probe. Never raise out
of a check, and never assert — report.

## Data handling

Every probe runs on seeded synthetic data generated in-process. Nothing reads
a recording, nothing writes outside `reports/`, and nothing contacts the
network. `--experiment-dir` is the one opt-in exception; even then the report
records shapes and counts, never samples.

The agent is fully deterministic and local. There is no model call in it, so
no code or data leaves the machine when it runs.
