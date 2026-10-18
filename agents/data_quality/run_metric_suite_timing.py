#!/usr/bin/env python3
"""Time full SPEC §4 metric families on one matched/lab session."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from analysis.real_quadrant.adapter import _parse_spike_times
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.chunks import chunk_bounds
from agents.data_quality.matching.regions import stratify_lab_units, add_mapped_lab_region
from agents.data_quality.metrics.guards import seconds_guard
from agents.data_quality.metrics.units import unit_metric_rows
from agents.data_quality.metrics.spatial import spatial_metric_rows
from agents.data_quality.metrics.population import population_metric_rows
from agents.data_quality.metrics.behavior_metrics import behavior_metric_rows
from agents.data_quality.metrics.rhythm import rhythm_metric_rows
from agents.data_quality.metrics.sorting_realism import sorting_metric_rows
from agents.data_quality.metrics.decoding import decoding_metric_rows

def main():
    session=os.environ.get("DQ_PILOT_SESSION","1150_27102024_2rooms")
    timings={}
    rows_all=[]
    # LAB
    bundle=build_segment_bundle_dq(session,"A")
    root=Path(os.environ["HIPPO_DATA_ROOT"])/session
    meta=pd.read_csv(root/"clusters_dataset.csv", usecols=lambda c: c in ("cell","Region","BClabel","timestamp"))
    meta=meta[meta["BClabel"].astype(str)!="NON-SOMA"].reset_index(drop=True)
    id_to={int(c):i for i,c in enumerate(meta["cell"])}
    spikes=[]; regions=[]
    for uid in bundle["unit_ids"]:
        i=id_to[int(uid)]; spikes.append(_parse_spike_times(meta.loc[i,"timestamp"])); regions.append(str(meta.loc[i,"Region"]))
    seconds_guard(spikes,name="lab")
    mask=stratify_lab_units(regions,set_name="primary")
    spikes=[spikes[i] for i in np.flatnonzero(mask)]
    regions=[regions[i] for i in np.flatnonzero(mask)]
    t0,t1=float(bundle["segment_t0"]),float(bundle["segment_t1"])
    bounds=chunk_bounds(t0,t1)
    lo,hi=bounds[0]
    # positions on decode grid
    times=np.asarray(bundle["decode_times"],float)
    y=np.asarray(bundle["y"],float)
    valid=np.asarray(bundle["target_valid"],bool)
    # speed from finite diff
    speed=np.zeros(len(times))
    for i in range(1,len(times)):
        if np.isfinite(y[i]).all() and np.isfinite(y[i-1]).all():
            speed[i]=np.linalg.norm(y[i]-y[i-1])/max(times[i]-times[i-1],1e-9)
    m=(times>=lo)&(times<hi)
    arena_hw=0.5*float(bundle.get("arena_width_cm") or bundle["arena_cm"][0] if "arena_cm" in bundle else 50)

    def time_fam(name, fn):
        t=time.perf_counter(); r=fn(); timings[name]=time.perf_counter()-t; rows_all.extend(r); return len(r)

    time_fam("units", lambda: unit_metric_rows(spikes,dataset="lab",animal_or_seed=session.split("_")[0],
        session=session,chunk=0,region_set="primary",t0=lo,t1=hi,regions=regions))
    time_fam("sorting", lambda: sorting_metric_rows(spikes,dataset="lab",animal_or_seed=session.split("_")[0],
        session=session,chunk=0,region_set="primary",t0=lo,t1=hi))
    time_fam("spatial", lambda: spatial_metric_rows(spikes,y[m],times[m],dataset="lab",
        animal_or_seed=session.split("_")[0],session=session,chunk=0,region_set="primary",
        t0=lo,t1=hi,speed=speed[m]))
    time_fam("rhythm", lambda: rhythm_metric_rows(spikes,dataset="lab",animal_or_seed=session.split("_")[0],
        session=session,chunk=0,region_set="primary"))
    time_fam("population", lambda: population_metric_rows(spikes,dataset="lab",animal_or_seed=session.split("_")[0],
        session=session,chunk=0,region_set="primary",t0=lo,t1=hi))
    time_fam("behavior", lambda: behavior_metric_rows(y[m][valid[m]], speed[m][valid[m]],
        dataset="lab",animal_or_seed=session.split("_")[0],session=session,chunk=0,region_set="primary",
        arena_half_width_cm=arena_hw))
    time_fam("decoding", lambda: decoding_metric_rows(spikes,y,times,dataset="lab",
        animal_or_seed=session.split("_")[0],session=session,chunk=0,region_set="primary",valid=valid,
        Ns=(10,min(20,len(spikes))),W_grid=(0.25,0.5)))

    # LDS eigenvalues — try import fit on small window
    t=time.perf_counter()
    try:
        from realtime.dynamic_latents.lds import fit_lds
        from agents.data_quality.metrics.population import _counts
        X=_counts(spikes, lo, hi, 0.25)
        if X.shape[0] > 50 and X.shape[1] >= 5:
            # subsample features
            Xs=X[:min(2000,len(X)), :min(20,X.shape[1])]
            model=fit_lds(Xs, n_latents=5) if 'fit_lds' in dir() else None
            timings["lds_eigen"]=time.perf_counter()-t
        else:
            timings["lds_eigen"]=time.perf_counter()-t
            timings["lds_eigen_note"]="skipped_small"
    except Exception as e:
        timings["lds_eigen"]=time.perf_counter()-t
        timings["lds_eigen_error"]=f"{type(e).__name__}: {e}"

    out={
        "session": session,
        "n_primary_units": len(spikes),
        "chunk": [lo,hi],
        "timings_s": timings,
        "total_s": float(sum(v for v in timings.values() if isinstance(v,(int,float)))),
        "n_rows": len(rows_all),
    }
    path=assert_dq_path(DATA_QUALITY_ROOT/"metrics_rows"/"suite_timing.json")
    path.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()
