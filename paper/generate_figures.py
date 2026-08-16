"""Generate aggregate, publication-safe figures for the manuscript."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/"figures"
OUT.mkdir(exist_ok=True)


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as handle: return list(csv.DictReader(handle))


def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight"); fig.savefig(OUT/f"{name}.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def am_figure():
    rows=read_csv(ROOT/"reports/track_a/experiment1_analysis_20260816_v2/am_quantitative_metrics.csv")
    methods=("raw","full_wave_rectified","hilbert_envelope","squared_energy"); snrs=("inf","20","10","0"); depths=("0.1","0.3","0.6")
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for method in methods:
        values=[100*np.mean([float(r["recovered"]) for r in rows if r["method"]==method and r["snr_db"]==snr]) for snr in snrs]
        axes[0].plot(snrs,values,marker="o",label=method.replace("_"," "))
    axes[0].set(xlabel="SNR (dB; inf = noise-free)",ylabel="Recovery (%)",ylim=(-3,103)); axes[0].grid(alpha=.2); axes[0].legend(fontsize=6)
    rect=[r for r in rows if r["method"]=="full_wave_rectified"]
    grid=np.array([[100*np.mean([float(r["recovered"]) for r in rect if r["modulation_depth"]==depth and r["snr_db"]==snr]) for snr in snrs] for depth in depths])
    image=axes[1].imshow(grid,aspect="auto",vmin=0,vmax=100,cmap="viridis")
    axes[1].set(xticks=range(4),xticklabels=snrs,yticks=range(3),yticklabels=depths,xlabel="SNR (dB)",ylabel="Modulation depth")
    for i in range(3):
        for j in range(4): axes[1].text(j,i,f"{grid[i,j]:.0f}",ha="center",va="center",color="white",fontsize=7)
    fig.colorbar(image,ax=axes[1],label="Recovery (%)",shrink=.8); save(fig,"am_mechanism")


def failure_figure():
    summary=json.load(open(ROOT/"reports/track_a/experiment2_failure_stages_20260816_v3/summary.json"))
    labels=("Representation","Fusion/scoring","Refinement","Mixed/unresolved")
    keys=("representation_failure","fusion_scoring_failure","refinement_failure","unresolved_mixed")
    counts=np.array([summary["failure_stages"][key]["n"] for key in keys]); percent=100*counts/counts.sum()
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    axes[0].bar(labels,percent,color=("#4477AA","#EE6677","#228833","#BBBBBB")); axes[0].set(ylabel="Material failures (%)",ylim=(0,42)); axes[0].tick_params(axis="x",rotation=25)
    for i,value in enumerate(percent): axes[0].text(i,value+.8,f"{value:.1f}%",ha="center",fontsize=7)
    rows=read_csv(ROOT/"reports/track_a/experiment2_failure_stages_20260816_v3/by_spectral_error_severity.csv")
    x=np.arange(len(rows)); axes[1].bar(x-.18,[float(r["mae"]) for r in rows],.36,label="spectral"); axes[1].bar(x+.18,[float(r["final_mae"]) for r in rows],.36,label="interval-refined")
    axes[1].set(xticks=x,xticklabels=[r["spectral_error_stratum"] for r in rows],xlabel="Spectral-error stratum (BPM)",ylabel="MAE (BPM)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.2,axis="y"); save(fig,"failure_mechanisms")


def degradation_figure():
    rows=read_csv(ROOT/"reports/track_a/experiment3_robustness_20260816_v2/paired_summary.csv")
    families=("gaussian_noise","clipping","contiguous_sample_loss"); severities=("mild","moderate","severe")
    specs=(("mean_delta_mean_target_to_distractor_ratio","$\\Delta$ target/distractor"),("mean_delta_mean_spectral_entropy","$\\Delta$ spectral entropy"),("mean_delta_cross_band_agreement","$\\Delta$ cross-band agreement"),("mean_delta_harmonic_family_stability","$\\Delta$ family stability"))
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.1)); colors=("#4477AA","#EE6677","#228833")
    for axis,(key,label) in zip(axes.ravel(),specs):
        for family,color in zip(families,colors):
            values=[float(next(r for r in rows if r["perturbation"]==family and r["severity"]==severity)[key]) for severity in severities]
            if key.endswith("harmonic_family_stability"): values=[value for value in values]
            axis.plot(range(1,4),values,marker="o",color=color,label=family.replace("_"," "))
        axis.axhline(0,color="black",linewidth=.7); axis.set(xticks=(1,2,3),xticklabels=severities,ylabel=label); axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=6); save(fig,"representation_degradation")


def reconstruction_figure():
    exp3=read_csv(ROOT/"reports/track_a/experiment3_robustness_20260816_v2/refinement_gate.csv")
    main=read_csv(ROOT/"reports/track_a/experiment4_nonlearned_20260816_v1/summary_by_condition.csv")
    corrected=read_csv(ROOT/"reports/track_a/experiment4_pchip_correction_20260816_v1/summary_by_condition.csv")
    main=[r for r in main if r["method"]!="pchip_interpolation"]+corrected
    conditions=(("gaussian_noise","bandlimit_0p5_12hz","Noise\nband-limit"),("gaussian_noise","wiener_5","Noise\nWiener"),("clipping","saturation_linear_interpolation","Clipping\ninterpolation"),("contiguous_sample_loss","linear_interpolation","Loss\nlinear"),("contiguous_sample_loss","pchip_interpolation","Loss\nPCHIP"))
    degraded=[]; recovered=[]
    triplets=read_csv(ROOT/"reports/track_a/experiment4_nonlearned_20260816_v1/triplets.csv"); corrected_triplets=read_csv(ROOT/"reports/track_a/experiment4_pchip_correction_20260816_v1/triplets.csv")
    triplets=[r for r in triplets if r["method"]!="pchip_interpolation"]+corrected_triplets
    for family,method,_ in conditions:
        selected=[r for r in triplets if r["perturbation"]==family and r["severity"]=="severe" and r["method"]==method]
        degraded.append(np.mean([float(r["degraded_final_error_bpm"]) for r in selected])); recovered.append(np.mean([float(r["final_error_bpm"]) for r in selected]))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8)); x=np.arange(len(conditions))
    axes[0].bar(x-.18,degraded,.36,label="degraded"); axes[0].bar(x+.18,recovered,.36,label="recovered"); axes[0].set(xticks=x,xticklabels=[c[2] for c in conditions],ylabel="MAE (BPM)"); axes[0].tick_params(axis="x",labelsize=6); axes[0].legend(fontsize=7); axes[0].grid(alpha=.2,axis="y")
    sample=triplets[::max(1,len(triplets)//15000)]; axes[1].scatter([float(r["waveform_recovery_fraction"]) for r in sample],[float(r["degraded_final_error_bpm"])-float(r["final_error_bpm"]) for r in sample],s=3,alpha=.12)
    axes[1].axhline(0,color="black",linewidth=.7); axes[1].set(xlabel="Waveform RMSE recovery fraction",ylabel="HR-error improvement (BPM)",xlim=(-3,1)); axes[1].grid(alpha=.2); save(fig,"reconstruction_task_mismatch")


if __name__=="__main__":
    am_figure(); failure_figure(); degradation_figure(); reconstruction_figure()
