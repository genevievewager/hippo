# Style inventory (Report 1 + Report 2) — used for Report 4a rebuild

Sources: `outputs/quadrant_n5/figures/quadrant_n5_figure_set.pdf`,
`outputs/real_quadrant/report2/report2_real_data.pdf`, and builders in
`agents/quadrant_n5/figures/{figures,figures_real,build_story,build_story_real}.py`.

## Story page
- ReportLab + Liberation Sans / Arial; A4; title → provenance strip → **The question** → numbered takeaways with **absolute numbers** → caveats / next.
- Numbers computed from tidy `data_*.csv` / `data_meta.json` (never hand-typed).
- Internal-draft footer; config/code hashes when available.

## Figure families (Report 1 map)
| Fig | Role |
| --- | --- |
| 1 Design | Schematic pipelines + population + example paths (not text boxes) |
| 2 Validity | PASS/FAIL audit matrix |
| 3 Answer | Primary result (paired / ranked) |
| 4 Mechanism | Why the answer holds |
| 5–7 | Behaviour / deployability / trajectories / summary |
| S1–S5 | Supplements |

## Panel style
- Double-column width `W_FULL = 180 mm`; lettered panels via `head` / `head_fig`.
- Ink palette: `INK`, `INK2`, `MUTED`, `GRID`, `FAILC`; method colours pca/dm/lds when relevant.
- Captions explain dots / lines / bands and **how examples were chosen**; n as animals/sessions/seeds.
- Overflow check before save (Report 2).

## Rate maps
- Train-only (or segment) maps; min occupancy stated; Gaussian smooth; Hz; equal aspect; arena outline.
- Unit selection by **fixed rule** (Report 4a: SI percentiles 25/50/75 within MEC and SUB — never top SI / peak rate).

## Trajectories
- Equal aspect; cm axes; arena outline; grey/black or start/end markers; coverage noted when relevant.
