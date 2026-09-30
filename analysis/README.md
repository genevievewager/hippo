# analysis/

Offline experiments and secondary workstreams.

This tree may import `hippo`, `hippo_sim`, `realtime`, and `visualization`.
Those packages must never import `analysis`.

Promote code into `hippo/` or `realtime/` only when a second consumer needs it
(see root `ARCHITECTURE.md`). Keep data and outputs outside git
(`HIPPO_DATA_ROOT`, local `outputs/`).
