# Quadrant N=5 figures (Phase 6)

Builds the publication figure set from a finished `quadrant_n5` run.

```bash
python agents/quadrant_n5/figures/make_figures.py \
    --results outputs/quadrant_n5 \
    --out outputs/quadrant_n5/figures
```

## What it reads

Only files under `--results`:

- `seed_*/{sorted,ground_truth}/source_summary.json`
- `seed_*/{sorted,ground_truth}/<method>.json` (these take precedence over the source summary when present)
- `seed_*/replay/sorted_summary.json`
- `audit_summary.json`
- `seed_*/sim/behavior.csv`

It never reads spikes, models, the probe-track YAML or anything outside `--results`, and it makes no network calls.

## What it writes

- `data/data_*.csv` and `data/data_meta.json`: the tidy source data behind every panel. Journals ask for these.
- `Fig1_design` to `Fig6_answer`: each as a vector `.pdf` and a 600-dpi `.png`, 180 mm wide (double column).
- `quadrant_n5_figure_set.pdf`: a story page, then one page per figure with its legend.

## Guarantees

- Every number in a legend or in the story page is computed from `data/`. None is typed by hand.
- It refuses to run if the results mix config hashes.
- Provenance comes from the results: config hash, the code SHA that produced the seeds, and the report SHA.
- A13 failures and the seed-4 diagnosis sentence are derived from the data. The figures stay correct on a re-run.

## Needs review on a new run

The story page's prose interprets this run: "dynamics is the axis that matters", "no static nonlinear advantage". The numbers inside it update automatically, but if the direction of an effect changes, the sentence around it will be wrong. Re-read page 1 after every new run.

## Dependencies

Uses numpy, pandas and matplotlib, plus reportlab and pypdf for the figure-set PDF only.

The fonts are Arial or Helvetica if installed, falling back to Liberation Sans and then DejaVu Sans. Fonts are embedded as TrueType (`pdf.fonttype 42`), so text stays editable in Illustrator.

## Encoding

- **Colour follows the quadrant cell:** blue for PCA, orange for DM, green for LDS. Controls and baselines are neutral grey.
- **Marker shape identifies the method**, so no information is carried by colour alone.
- **Filled vs. hollow:**
  - sorted spikes vs. ground-truth spikes
  - Ridge vs. kNN
- **GPFA is always hollow and marked †:** it is offline-only.

The three-hue palette passed a colour-vision-deficiency (CVD) check across all pairs of colours.
