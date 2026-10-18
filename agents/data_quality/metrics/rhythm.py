"""Spike-based theta modulation from autocorrelogram."""
from __future__ import annotations
from typing import Any
import numpy as np
from agents.data_quality.metrics.schema import tidy_row

def theta_modulation_index(st, max_lag=0.5, bin_s=0.01):
    st=np.sort(np.asarray(st,float))
    if st.size < 20:
        return float("nan"), float("nan")
    # pairwise diffs subsample
    if st.size > 2000:
        st=st[:: max(1, st.size//2000)]
    d=[]
    for i,t in enumerate(st):
        j=np.searchsorted(st, t+max_lag)
        dd=st[i+1:j]-t
        d.append(dd)
    if not d:
        return float("nan"), float("nan")
    d=np.concatenate(d)
    bins=np.arange(bin_s, max_lag+bin_s, bin_s)
    acg,_=np.histogram(d, bins=bins)
    # 6-10 Hz peak vs baseline 10-50Hz-ish in lag domain: period 0.1-0.167s
    centers=0.5*(bins[:-1]+bins[1:])
    theta=(centers>=0.1)&(centers<=0.167)
    base=(centers>=0.05)&(centers<=0.5)&(~theta)
    if not theta.any() or not base.any() or acg[base].mean()<=0:
        return float("nan"), float("nan")
    tmi=(acg[theta].mean()-acg[base].mean())/acg[base].mean()
    # peak freq
    if acg[theta].sum()==0:
        return float(tmi), float("nan")
    peak=centers[theta][int(np.argmax(acg[theta]))]
    return float(tmi), float(1.0/peak)

def rhythm_metric_rows(spike_times, *, dataset, animal_or_seed, session, chunk, region_set):
    tmis=[]; freqs=[]
    for st in spike_times:
        tmi,f=theta_modulation_index(st)
        tmis.append(tmi); freqs.append(f)
    base=dict(dataset=dataset, animal_or_seed=animal_or_seed, session=session, chunk=chunk, region_set=region_set)
    return [
        tidy_row(**base, metric="theta_mod_index_median", W=None, N=None, value=float(np.nanmedian(tmis))),
        tidy_row(**base, metric="theta_freq_median", W=None, N=None, value=float(np.nanmedian(freqs))),
    ]
