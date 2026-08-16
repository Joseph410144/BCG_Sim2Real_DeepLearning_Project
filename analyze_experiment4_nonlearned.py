"""Paired non-learned recovery baselines for Experiment 4."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import wiener
from scipy.stats import spearmanr

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.Filters import BandPassFilter
from analyze_experiment2_failure_stages import analyze_signal, read_csv, write_csv
from analyze_experiment3_robustness import perturb, result_row, stable_rng


FEATURES=("mean_target_to_distractor_ratio","min_target_to_distractor_ratio",
          "mean_spectral_entropy","cross_band_agreement","harmonic_family_stability",
          "spectral_error_bpm","final_error_bpm")


def interpolation(values, missing, method):
    known=np.flatnonzero(~missing); targets=np.flatnonzero(missing)
    if not len(targets) or len(known)<2: return values.copy()
    result=values.copy()
    if method=="linear_interpolation": result[targets]=np.interp(targets,known,values[known])
    elif method=="pchip_interpolation":
        result[targets]=np.interp(targets,known,values[known])
        interior=targets[(targets>known[0]) & (targets<known[-1])]
        if len(interior): result[interior]=PchipInterpolator(known,values[known],extrapolate=False)(interior)
    else: raise ValueError(method)
    return result


def recover(degraded, family, method, spec, clean):
    if method=="bandlimit_0p5_12hz": return BandPassFilter(degraded,.5,12,4,100,padlen=500)
    if method=="wiener_5": return np.asarray(wiener(degraded,mysize=5),dtype=float)
    if family=="clipping":
        threshold=spec["threshold_sd"]*np.std(clean); saturated=np.abs(degraded)>=threshold-1e-10
        return interpolation(degraded,saturated,"linear_interpolation")
    if family=="contiguous_sample_loss":
        missing=np.zeros(len(degraded),dtype=bool); missing[np.isclose(degraded,0,atol=1e-14)]=True
        return interpolation(degraded,missing,method)
    raise ValueError((family,method))


def normalized_recovery(clean,degraded,recovered,epsilon):
    denominator=clean-degraded
    return np.nan if abs(denominator)<epsilon else (recovered-degraded)/denominator


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row["perturbation"],row["severity"],row["method"])].append(row)
    output=[]
    for (family,severity,method),selected in sorted(groups.items()):
        entry={"perturbation":family,"severity":severity,"method":method,"n":len(selected),
               "waveform_rmse":np.mean([float(r["recovered_waveform_rmse"]) for r in selected]),
               "waveform_recovery_fraction":np.mean([float(r["waveform_recovery_fraction"]) for r in selected]),
               "final_mae":np.mean([float(r["final_error_bpm"]) for r in selected]),
               "spectral_mae":np.mean([float(r["spectral_error_bpm"]) for r in selected]),
               "within_5_percent":100*np.mean([float(r["final_error_bpm"])<=5 for r in selected]),
               "material_failure_percent":100*np.mean([float(r["final_error_bpm"])>5 for r in selected])}
        for feature in FEATURES:
            valid=[float(r[f"recovery_fraction_{feature}"]) for r in selected if r[f"recovery_fraction_{feature}"]!=""]
            entry[f"mean_recovery_fraction_{feature}"]=np.mean(valid) if valid else np.nan
            entry[f"median_recovery_fraction_{feature}"]=np.median(valid) if valid else np.nan
            entry[f"mean_remaining_distance_{feature}"]=np.mean([abs(float(r[feature])-float(r[f"clean_{feature}"])) for r in selected])
        output.append(entry)
    return output


def source_aggregate(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row["continuous_source_id"],row["perturbation"],row["severity"],row["method"])].append(row)
    return [{"continuous_source_id":key[0],"perturbation":key[1],"severity":key[2],"method":key[3],"n":len(selected),
             "waveform_recovery_fraction":np.mean([float(r["waveform_recovery_fraction"]) for r in selected]),
             "final_mae":np.mean([float(r["final_error_bpm"]) for r in selected]),
             "degraded_final_mae":np.mean([float(r["degraded_final_error_bpm"]) for r in selected]),
             "final_error_improvement":np.mean([float(r["degraded_final_error_bpm"])-float(r["final_error_bpm"]) for r in selected]),
             "td_improvement":np.mean([float(r["mean_target_to_distractor_ratio"])-float(r["degraded_mean_target_to_distractor_ratio"]) for r in selected])}
            for key,selected in groups.items()]


def plot_results(rows,summary,output):
    methods=sorted({r["method"] for r in summary}); severities=("mild","moderate","severe")
    fig,axes=plt.subplots(2,3,figsize=(14,8)); axes=axes.ravel(); plot_features=("mean_target_to_distractor_ratio","mean_spectral_entropy","cross_band_agreement","harmonic_family_stability","final_error_bpm")
    for axis,feature in zip(axes,plot_features):
        for method in methods:
            values=[]
            for severity in severities:
                selected=[r for r in summary if r["method"]==method and r["severity"]==severity]
                values.append(np.mean([float(r[f"median_recovery_fraction_{feature}"]) for r in selected]) if selected else np.nan)
            axis.plot(range(3),values,marker="o",label=method)
        axis.axhline(1,color="k",linestyle="--",linewidth=.8); axis.axhline(0,color="gray",linestyle=":",linewidth=.8)
        axis.set(xticks=range(3),xticklabels=severities,title=feature.replace("_"," "),ylabel="Median normalized recovery"); axis.grid(alpha=.2)
    axes[5].axis("off"); axes[5].legend(*axes[0].get_legend_handles_labels(),loc="center",fontsize=7)
    fig.tight_layout(); fig.savefig(output/"normalized_feature_recovery.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sample=rows[::max(1,len(rows)//20000)]
    x=np.asarray([float(r["waveform_recovery_fraction"]) for r in sample])
    axes[0].scatter(x,[float(r["degraded_final_error_bpm"])-float(r["final_error_bpm"]) for r in sample],s=3,alpha=.15)
    axes[0].set(xlabel="Waveform RMSE recovery fraction",ylabel="Final HR error improvement (BPM)")
    axes[1].scatter(x,[float(r["mean_target_to_distractor_ratio"])-float(r["degraded_mean_target_to_distractor_ratio"]) for r in sample],s=3,alpha=.15)
    axes[1].set(xlabel="Waveform RMSE recovery fraction",ylabel="T/D improvement")
    for axis in axes: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"waveform_vs_task_recovery.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--robustness-config",type=Path,required=True)
    parser.add_argument("--experiment3-pairs",type=Path,required=True); parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--data-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); config=json.loads(args.config.read_text()); robust=json.loads(args.robustness_config.read_text()); args.output_dir.mkdir(parents=True,exist_ok=False)
    exp3={(r["source_filename"],r["perturbation"],r["severity"]):r for r in read_csv(args.experiment3_pairs) if r["severity"]!="clean"}
    baseline={r["filename"]:r for r in read_csv(args.baseline)}; rows=[]; manifest=read_csv(args.manifest)
    for index,provenance in enumerate(manifest):
        name=provenance["source_filename"]; ecg=float(baseline[name]["ecg_hr"]); pair=np.load(args.data_dir/name)
        clean=zscore_normalize(BandPassFilter(pair[0],.5,25,4,100,padlen=500)); clean_result=analyze_signal(clean,ecg); clean_row,clean_families=result_row(clean_result,ecg)
        for family in config["methods"]:
            specs=robust["perturbations"][family]
            for spec in specs:
                severity=spec["label"]; rng=stable_rng(config["seed"],name,family,severity); degraded=perturb(clean,family,spec,rng); degraded_row=exp3[(name,family,severity)]
                degraded_rmse=float(np.sqrt(np.mean((degraded-clean)**2)))
                for method in config["methods"][family]:
                    recovered=recover(degraded,family,method,spec,clean); recovered_result=analyze_signal(recovered,ecg); recovered_row,_=result_row(recovered_result,ecg,clean_families)
                    row={"source_filename":name,"continuous_source_id":provenance["continuous_source_id"],"legacy_dataset_id":provenance["legacy_dataset_id"],"perturbation":family,"severity":severity,"method":method,
                         "degraded_waveform_rmse":degraded_rmse,"recovered_waveform_rmse":float(np.sqrt(np.mean((recovered-clean)**2)))}
                    row["waveform_recovery_fraction"]=(degraded_rmse-row["recovered_waveform_rmse"])/(degraded_rmse+1e-12)
                    for feature in FEATURES:
                        row[f"clean_{feature}"]=clean_row[feature]; row[f"degraded_{feature}"]=degraded_row[feature]; row[feature]=recovered_row[feature]
                        value=normalized_recovery(float(clean_row[feature]),float(degraded_row[feature]),float(recovered_row[feature]),config["normalization_denominator_epsilon"])
                        row[f"recovery_fraction_{feature}"]="" if np.isnan(value) else value
                    row["degraded_failure_stage"]=degraded_row["failure_stage"]; row["recovered_failure_stage"]=recovered_row["failure_stage"]
                    rows.append(row)
        if (index+1)%100==0: print(f"processed {index+1}/{len(manifest)}",flush=True)
    write_csv(args.output_dir/"triplets.csv",rows); summary=aggregate(rows); write_csv(args.output_dir/"summary_by_condition.csv",summary); sources=source_aggregate(rows); write_csv(args.output_dir/"source_summary.csv",sources)
    correlations={}
    for outcome in ("final_error_improvement","td_improvement"):
        y=[float(r["degraded_final_error_bpm"])-float(r["final_error_bpm"]) for r in rows] if outcome=="final_error_improvement" else [float(r["mean_target_to_distractor_ratio"])-float(r["degraded_mean_target_to_distractor_ratio"]) for r in rows]
        rho,p=spearmanr([float(r["waveform_recovery_fraction"]) for r in rows],y); correlations[outcome]={"spearman_rho":float(rho),"p_value_descriptive_only":float(p)}
    transitions=[]
    counts=defaultdict(int)
    for r in rows: counts[(r["perturbation"],r["severity"],r["method"],r["degraded_failure_stage"],r["recovered_failure_stage"])] += 1
    for key,n in counts.items(): transitions.append({"perturbation":key[0],"severity":key[1],"method":key[2],"from_stage":key[3],"to_stage":key[4],"n":n})
    write_csv(args.output_dir/"failure_transitions.csv",transitions); plot_results(rows,summary,args.output_dir)
    result={"dataset_interpretation":"single_person_repeated_recordings","clean_windows":len(manifest),"triplets":len(rows),"methods":config["methods"],"waveform_task_correlations":correlations}
    (args.output_dir/"summary.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
