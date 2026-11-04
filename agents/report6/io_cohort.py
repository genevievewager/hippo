"""Load Report 2 / Report 1 caches with S01–S10 aliases (IDs only in gitignored map)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from agents.report6.paths import (
    ALIAS_PATH,
    OUT_ROOT,
    R1_MATCHED_SIM,
    R2_COUNT_CACHE,
    R2_ROOM_A,
)


ANIMALS = list("ABCDEF")


@dataclass
class SessionData:
    alias: str  # S01…
    animal_alias: str  # A…
    arm: str  # real | sim
    n_units: int
    primary_d_lds: int
    primary_d_pca: int
    Z_lds: np.ndarray
    Z_pca: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_ok: np.ndarray
    retained: np.ndarray
    valid: np.ndarray
    decode_times: np.ndarray
    y: np.ndarray | None
    X_counts: np.ndarray | None
    source_dir: str  # absolute; not for captions


def _session_dirs_real() -> list[Path]:
    return sorted(p for p in R2_ROOM_A.iterdir() if p.is_dir() and (p / "figure_contract").is_dir())


def _session_dirs_sim_k0() -> list[Path]:
    out = []
    for sess in sorted(R1_MATCHED_SIM.iterdir()):
        sub = sess / "subsample_00"
        if (sub / "figure_contract" / "latents.npz").is_file():
            out.append(sub)
    return out


def _animal_key_from_dirname(name: str) -> str:
    # dirname starts with animal digits before first _; never print this
    return name.split("_", 1)[0]


def build_alias_map(force: bool = False) -> dict[str, Any]:
    """Write gitignored alias map; return public view without real IDs if possible."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if ALIAS_PATH.is_file() and not force:
        return json.loads(ALIAS_PATH.read_text())

    real_dirs = _session_dirs_real()
    sim_dirs = _session_dirs_sim_k0()
    animal_order: list[str] = []
    animal_alias: dict[str, str] = {}
    rows = []
    for i, p in enumerate(real_dirs, 1):
        akey = _animal_key_from_dirname(p.name)
        if akey not in animal_alias:
            animal_alias[akey] = ANIMALS[len(animal_order) % len(ANIMALS)]
            animal_order.append(akey)
        rows.append({
            "alias": f"S{i:02d}",
            "animal_alias": animal_alias[akey],
            "arm": "real",
            "dirname": p.name,
            "path": str(p),
            "path_sha12": hashlib.sha256(str(p).encode()).hexdigest()[:12],
        })
    for j, p in enumerate(sim_dirs, 1):
        # match real order by parent session dirname
        parent = p.parent.name
        akey = _animal_key_from_dirname(parent)
        aa = animal_alias.get(akey, ANIMALS[(j - 1) % len(ANIMALS)])
        rows.append({
            "alias": f"SIM{j:02d}",
            "animal_alias": aa,
            "arm": "sim",
            "dirname": parent,
            "path": str(p),
            "path_sha12": hashlib.sha256(str(p).encode()).hexdigest()[:12],
            "subsample_k": 0,
        })
    blob = {"n_real": len(real_dirs), "n_sim": len(sim_dirs), "rows": rows}
    ALIAS_PATH.write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def _load_primary_d(latent_cache: Path, method: str) -> int:
    z = np.load(latent_cache / "final" / f"{method}.npz")
    return int(z["primary_d"][0])


def _match_x_counts(T: int, n_units: int) -> np.ndarray | None:
    for p in sorted(R2_COUNT_CACHE.glob("*.npz")):
        z = np.load(p)
        X = z["X_counts"]
        if X.shape == (T, n_units):
            return np.asarray(X, dtype=float)
    # T match only, take n_units columns if wider (should not happen for HF)
    for p in sorted(R2_COUNT_CACHE.glob("*.npz")):
        z = np.load(p)
        X = np.asarray(z["X_counts"], dtype=float)
        if X.shape[0] == T and X.shape[1] >= n_units:
            # Prefer exact; skip ambiguous
            continue
    return None


def load_session(row: dict[str, Any]) -> SessionData:
    root = Path(row["path"])
    fc = root / "figure_contract"
    lc = root / "latent_cache"
    lat = np.load(fc / "latents.npz", allow_pickle=True)
    arr = np.load(fc / "arrays.npz", allow_pickle=True)
    meta = json.loads((lc / "meta.json").read_text()) if (lc / "meta.json").is_file() else {}
    n_units = int(meta.get("n_units") or lat["Z_raw"].shape[1] if "Z_raw" in lat.files else -1)

    if (lc / "final" / "lds.npz").is_file():
        d_lds = _load_primary_d(lc, "lds")
        d_pca = _load_primary_d(lc, "pca")
        Z_lds = np.asarray(np.load(lc / "final" / "lds.npz")["Z"], dtype=float)[:, :d_lds]
        Z_pca = np.asarray(np.load(lc / "final" / "pca.npz")["Z"], dtype=float)[:, :d_pca]
    else:
        Z_lds_full = np.asarray(lat["Z_lds"], dtype=float)
        Z_pca_full = np.asarray(lat["Z_pca"], dtype=float)
        d_lds = Z_lds_full.shape[1]
        d_pca = Z_pca_full.shape[1]
        Z_lds, Z_pca = Z_lds_full, Z_pca_full

    X = None
    if "X_counts" in arr.files:
        X = np.asarray(arr["X_counts"], dtype=float)
        n_units = int(X.shape[1])
    else:
        if n_units < 0:
            n_units = int(json.loads((lc / "meta.json").read_text())["n_units"])
        X = _match_x_counts(int(arr["train_mask"].shape[0]), n_units)

    y = np.asarray(arr["y"], dtype=float) if "y" in arr.files else None
    return SessionData(
        alias=row["alias"],
        animal_alias=row["animal_alias"],
        arm=row["arm"],
        n_units=int(n_units),
        primary_d_lds=int(d_lds),
        primary_d_pca=int(d_pca),
        Z_lds=Z_lds,
        Z_pca=Z_pca,
        train_mask=np.asarray(arr["train_mask"], dtype=bool),
        test_mask=np.asarray(arr["test_mask"], dtype=bool),
        train_ok=np.asarray(arr["train_ok"], dtype=bool),
        retained=np.asarray(arr["retained"], dtype=bool),
        valid=np.asarray(arr["valid"], dtype=bool),
        decode_times=np.asarray(arr["decode_times"], dtype=float),
        y=y,
        X_counts=X,
        source_dir=str(root),
    )


def iter_sessions(arm: str | None = None) -> Iterator[SessionData]:
    amap = build_alias_map()
    for row in amap["rows"]:
        if arm is not None and row["arm"] != arm:
            continue
        yield load_session(row)


def public_cohort_table() -> list[dict[str, Any]]:
    """Counts-only table safe for captions / chat."""
    rows = []
    for s in iter_sessions():
        rows.append({
            "alias": s.alias,
            "animal": s.animal_alias,
            "arm": s.arm,
            "n_units": s.n_units,
            "d_lds": s.primary_d_lds,
            "d_pca": s.primary_d_pca,
            "T": int(s.train_mask.size),
            "n_train": int(s.train_ok.sum()),
            "n_test": int(s.test_mask.sum()),
            "has_X": s.X_counts is not None,
        })
    return rows
