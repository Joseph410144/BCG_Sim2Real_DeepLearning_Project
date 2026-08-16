"""Paired representation-focused robustness analysis for HeartV6."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.Filters import BandPassFilter
from analyze_experiment2_failure_stages import analyze_signal, classify_stage, metrics, read_csv, write_csv


SEVERITY_ORDER = {"clean": 0, "mild": 1, "moderate": 2, "severe": 3}
FEATURES = ("mean_target_to_distractor_ratio", "min_target_to_distractor_ratio",
            "mean_spectral_entropy", "cross_band_agreement", "ecg_target_band_agreement",
            "harmonic_family_stability")


def stable_rng(seed, filename, perturbation, severity):
    digest = hashlib.sha256(f"{seed}|{filename}|{perturbation}|{severity}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def perturb(values, family, spec, rng):
    if family == "gaussian_noise":
        power = np.mean(values**2); noise_power = power/(10**(spec["snr_db"]/10))
        return values + rng.normal(0, np.sqrt(noise_power), len(values))
    if family == "amplitude_attenuation": return values*spec["gain"]
    if family == "clipping":
        threshold = spec["threshold_sd"]*np.std(values)
        return np.clip(values, -threshold, threshold)
    if family == "contiguous_sample_loss":
        result = values.copy(); length = max(1, int(round(len(values)*spec["fraction"])))
        start = int(rng.integers(0, len(values)-length+1)); result[start:start+length] = 0
        return result
    raise ValueError(family)


def family_label(frequency, target):
    distances = {"half": abs(frequency-.5*target), "fundamental": abs(frequency-target),
                 "double": abs(frequency-2*target)}
    label = min(distances, key=distances.get)
    return label if distances[label] <= .15 else "other"


def result_row(result, ecg, clean_families=None):
    target=ecg/60; spectral=60*result["current_selected"]; final=result["final_hr"]
    per_band=[family_label(value,target) for value in result["preferences"]]
    fused=family_label(result["current_selected"],target); families=per_band+[fused]
    stability=1.0 if clean_families is None else float(np.mean(np.asarray(families)==np.asarray(clean_families)))
    row={"ecg_hr":ecg,"selected_frequency_hz":result["current_selected"],"spectral_hr":spectral,"final_hr":final,
         "spectral_error_bpm":abs(spectral-ecg),"final_error_bpm":abs(final-ecg),
         "mean_target_to_distractor_ratio":result["mean_td"],"min_target_to_distractor_ratio":result["min_td"],
         "mean_spectral_entropy":result["mean_entropy"],"cross_band_agreement":result["cross_agreement"],
         "ecg_target_band_agreement":result["target_agreement"],"per_band_preferred_harmonic_family":"|".join(per_band),
         "fused_selected_family":fused,"harmonic_family_stability":stability,"interval_cv":result["interval_cv"],
         "spectral_interval_disagreement_bpm":abs(final-spectral),"production_score_margin":result["score_margin"],
         "peak_count":result["peak_count"]}
    row["failure_stage"]=classify_stage({**row,"refinement_effect_bpm":row["final_error_bpm"]-row["spectral_error_bpm"],
                                          "argmax_hr":60*result["argmax_selected"],
                                          "normalized_score_hr":60*result["normalized_selected"],
                                          "modal_preferred_frequency_hz":result["modal_preferred"]})
    return row, families


def paired_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        if row["severity"] != "clean": groups[(row["perturbation"],row["severity"])].append(row)
    output=[]
    for (family,severity), selected in sorted(groups.items(),key=lambda x:(x[0][0],SEVERITY_ORDER[x[0][1]])):
        entry={"perturbation":family,"severity":severity,"n":len(selected)}
        for feature in FEATURES+("final_error_bpm","spectral_error_bpm","interval_cv","spectral_interval_disagreement_bpm","production_score_margin"):
            changes=np.asarray([float(row[feature])-float(row[f"clean_{feature}"]) for row in selected])
            entry[f"mean_delta_{feature}"]=float(np.mean(changes)); entry[f"median_delta_{feature}"]=float(np.median(changes))
        entry["new_material_failure_percent"]=100*np.mean([float(r["clean_final_error_bpm"])<=5 and float(r["final_error_bpm"])>5 for r in selected])
        entry["gate_mae"]=float(np.mean([float(r["gated_error_bpm"]) for r in selected]))
        entry["final_mae"]=float(np.mean([float(r["final_error_bpm"]) for r in selected]))
        output.append(entry)
    return output


def source_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        if row["severity"]!="clean": groups[(row["continuous_source_id"],row["perturbation"],row["severity"])].append(row)
    output=[]
    for (source,family,severity), selected in groups.items():
        output.append({"continuous_source_id":source,"perturbation":family,"severity":severity,"n":len(selected),
                       "mean_delta_td":np.mean([float(r["mean_target_to_distractor_ratio"])-float(r["clean_mean_target_to_distractor_ratio"]) for r in selected]),
                       "mean_delta_entropy":np.mean([float(r["mean_spectral_entropy"])-float(r["clean_mean_spectral_entropy"]) for r in selected]),
                       "mean_delta_cross_band_agreement":np.mean([float(r["cross_band_agreement"])-float(r["clean_cross_band_agreement"]) for r in selected]),
                       "mean_family_stability":np.mean([float(r["harmonic_family_stability"]) for r in selected]),
                       "final_mae":np.mean([float(r["final_error_bpm"]) for r in selected]),
                       "gated_mae":np.mean([float(r["gated_error_bpm"]) for r in selected])})
    return output


def transition_rows(rows):
    counts=Counter((r["perturbation"],r["severity"],r["clean_failure_stage"],r["failure_stage"])
                   for r in rows if r["severity"]!="clean")
    return [{"perturbation":k[0],"severity":k[1],"from_stage":k[2],"to_stage":k[3],"n":v}
            for k,v in sorted(counts.items())]


def plot_aggregates(rows, transitions, output):
    summary=paired_summary(rows); families=sorted({r["perturbation"] for r in summary})
    plot_specs=(("mean_target_to_distractor_ratio","T/D"),("mean_spectral_entropy","Spectral entropy"),
                ("cross_band_agreement","Cross-band agreement"),("harmonic_family_stability","Family stability"),
                ("final_error_bpm","Final HR error (BPM)"))
    fig,axes=plt.subplots(2,3,figsize=(14,8)); axes=axes.ravel()
    for axis,(feature,label) in zip(axes,plot_specs):
        for family in families:
            selected=[r for r in rows if r["perturbation"]==family]
            means=[]
            for severity in ("clean","mild","moderate","severe"):
                values=[float(r[feature]) for r in rows if r["severity"]=="clean"] if severity == "clean" else [float(r[feature]) for r in selected if r["severity"]==severity]
                means.append(np.mean(values))
            axis.plot(range(4),means,marker="o",label=family)
        axis.set(xticks=range(4),xticklabels=("clean","mild","moderate","severe"),ylabel=label); axis.grid(alpha=.2)
    axes[5].axis("off"); axes[5].legend(*axes[0].get_legend_handles_labels(),loc="center")
    fig.tight_layout(); fig.savefig(output/"severity_feature_curves.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(13,8));
    for axis,feature in zip(axes.ravel(),FEATURES):
        clean=np.asarray([float(r[f"clean_{feature}"]) for r in rows if r["severity"]!="clean"])
        degraded=np.asarray([float(r[feature]) for r in rows if r["severity"]!="clean"])
        axis.scatter(clean,degraded,s=2,alpha=.08); lo=min(clean.min(),degraded.min()); hi=max(clean.max(),degraded.max())
        axis.plot([lo,hi],[lo,hi],"k--"); axis.set(xlabel="Clean",ylabel="Perturbed",title=feature.replace("_"," "))
    fig.tight_layout(); fig.savefig(output/"clean_vs_perturbed_features.png",dpi=180); plt.close(fig)

    stages=sorted({r["from_stage"] for r in transitions}|{r["to_stage"] for r in transitions})
    fig,axes=plt.subplots(2,2,figsize=(12,10));
    for axis,family in zip(axes.ravel(),families):
        matrix=np.zeros((len(stages),len(stages)))
        for r in transitions:
            if r["perturbation"]==family: matrix[stages.index(r["from_stage"]),stages.index(r["to_stage"])]+=int(r["n"])
        matrix=matrix/(matrix.sum(axis=1,keepdims=True)+1e-12)*100
        image=axis.imshow(matrix,cmap="magma"); axis.set(xticks=range(len(stages)),xticklabels=stages,yticks=range(len(stages)),yticklabels=stages,title=family,xlabel="Perturbed stage",ylabel="Clean stage")
        axis.tick_params(axis="x",rotation=60,labelsize=6); axis.tick_params(axis="y",labelsize=6); fig.colorbar(image,ax=axis,shrink=.7)
    fig.tight_layout(); fig.savefig(output/"failure_stage_transitions.png",dpi=180); plt.close(fig)


def common_axis(rows):
    degraded=[r for r in rows if r["severity"]!="clean"]
    keys=("mean_target_to_distractor_ratio","mean_spectral_entropy","cross_band_agreement","harmonic_family_stability")
    matrix=np.asarray([[float(r[k])-float(r[f"clean_{k}"]) for k in keys] for r in degraded])
    scale=np.std(matrix,axis=0); standardized=(matrix-np.mean(matrix,axis=0))/(scale+1e-12)
    _,singular,vh=np.linalg.svd(standardized,full_matrices=False)
    variance=singular**2/np.sum(singular**2)
    return {"features":keys,"pc1_explained_percent":float(100*variance[0]),"pc1_loadings":dict(zip(keys,map(float,vh[0])))}


def plot_representative_trajectories(rows, output):
    families=sorted({r["perturbation"] for r in rows if r["severity"]!="clean"})
    fig,axes=plt.subplots(2,2,figsize=(12,9)); trajectory_features=(
        "mean_target_to_distractor_ratio","mean_spectral_entropy",
        "cross_band_agreement","harmonic_family_stability")
    for axis,family in zip(axes.ravel(),families):
        severe=[r for r in rows if r["perturbation"]==family and r["severity"]=="severe"]
        chosen=min(severe,key=lambda r:float(r["mean_target_to_distractor_ratio"])-float(r["clean_mean_target_to_distractor_ratio"]))
        selected=[r for r in rows if r["source_filename"]==chosen["source_filename"] and
                  ((r["perturbation"]==family) or r["severity"]=="clean")]
        for feature in trajectory_features:
            values=[]
            for severity in ("clean","mild","moderate","severe"):
                match=next(r for r in selected if r["severity"]==severity)
                values.append(float(match[feature]))
            normalized=np.asarray(values)/(abs(values[0])+1e-12)
            axis.plot(range(4),normalized,marker="o",label=feature.replace("_"," "))
        axis.set(title=family,xticks=range(4),xticklabels=("clean","mild","moderate","severe"),ylabel="Value / clean value")
        axis.grid(alpha=.2); axis.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output/"representative_feature_trajectories.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    config=json.loads(args.config.read_text()); args.output_dir.mkdir(parents=True,exist_ok=False)
    baseline={r["filename"]:r for r in read_csv(args.baseline)}; manifest=read_csv(args.manifest); rows=[]
    for index,provenance in enumerate(manifest):
        name=provenance["source_filename"]; ecg=float(baseline[name]["ecg_hr"]); pair=np.load(args.data_dir/name)
        clean_signal=zscore_normalize(BandPassFilter(pair[0],.5,25,4,100,padlen=500)); clean_result=analyze_signal(clean_signal,ecg)
        clean_row,clean_families=result_row(clean_result,ecg); clean_row.update({"source_filename":name,"continuous_source_id":provenance["continuous_source_id"],"legacy_dataset_id":provenance["legacy_dataset_id"],"perturbation":"clean","severity":"clean"})
        clean_row["gated_hr"]=clean_row["spectral_hr"] if clean_row["interval_cv"]>config["refinement_gate_interval_cv"] else clean_row["final_hr"]
        clean_row["gated_error_bpm"]=abs(clean_row["gated_hr"]-ecg)
        clean_fields={f"clean_{key}":value for key,value in clean_row.items() if key not in ("source_filename","continuous_source_id","legacy_dataset_id")}
        clean_row.update(clean_fields); rows.append(clean_row)
        for family,specs in config["perturbations"].items():
            for spec in specs:
                rng=stable_rng(config["seed"],name,family,spec["label"]); degraded=perturb(clean_signal,family,spec,rng)
                result=analyze_signal(degraded,ecg); row,_=result_row(result,ecg,clean_families)
                row.update({"source_filename":name,"continuous_source_id":provenance["continuous_source_id"],"legacy_dataset_id":provenance["legacy_dataset_id"],"perturbation":family,"severity":spec["label"],**clean_fields})
                row["gated_hr"]=row["spectral_hr"] if row["interval_cv"]>config["refinement_gate_interval_cv"] else row["final_hr"]
                row["gated_error_bpm"]=abs(row["gated_hr"]-ecg); rows.append(row)
        if (index+1)%100==0: print(f"processed {index+1}/{len(manifest)}",flush=True)
    write_csv(args.output_dir/"paired_windows.csv",rows); summary_rows=paired_summary(rows); write_csv(args.output_dir/"paired_summary.csv",summary_rows)
    sources=source_summary(rows); write_csv(args.output_dir/"source_summary.csv",sources); transitions=transition_rows(rows); write_csv(args.output_dir/"failure_transitions.csv",transitions)
    gate=[]
    for condition in [("clean","clean")]+[(f,s) for f in config["perturbations"] for s in ("mild","moderate","severe")]:
        selected=[r for r in rows if r["perturbation"]==condition[0] and r["severity"]==condition[1]]
        gate.append({"perturbation":condition[0],"severity":condition[1],"n":len(selected),"final_mae":np.mean([float(r["final_error_bpm"]) for r in selected]),"gated_mae":np.mean([float(r["gated_error_bpm"]) for r in selected]),"gate_activation_percent":100*np.mean([float(r["interval_cv"])>config["refinement_gate_interval_cv"] for r in selected])})
    write_csv(args.output_dir/"refinement_gate.csv",gate); plot_aggregates(rows,transitions,args.output_dir)
    plot_representative_trajectories(rows,args.output_dir)
    result={"dataset_interpretation":"single_person_repeated_recordings","clean_windows":len(manifest),"paired_degraded_windows":len(rows)-len(manifest),"config":config,"common_representation_axis":common_axis(rows),"gate":gate}
    (args.output_dir/"summary.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
