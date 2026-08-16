"""Full-dataset, read-only HeartV6 failure-stage analysis."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from Algorithm.Data_pre_processing import zscore_normalize
from Algorithm.Filters import BandPassFilter
from Algorithm.heart_v6_trace import HEART_V6_BANDS


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def window_energy(frequencies, spectrum, center, half_width=0.15):
    return float(np.sum(spectrum[np.abs(frequencies-center) <= half_width]))


def candidate_scores(frequencies, fused, min_hz=.7, max_hz=3.0):
    candidates = frequencies[(frequencies > min_hz) & (frequencies <= max_hz)]
    total, count = [], []
    for candidate in candidates:
        terms = [window_energy(frequencies, fused, harmonic*candidate)
                 for harmonic in range(1, 20) if harmonic*candidate < max_hz]
        total.append(sum(terms)); count.append(len(terms))
    total, count = np.asarray(total), np.asarray(count)
    current = total / (count + 1)  # exact production divisor after loop increment
    normalized = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return candidates, total, count, current, normalized


def select_from_family(base, frequencies, fused, max_hz=3.0):
    options, scores = [], []
    harmonic = 1
    while harmonic*base < max_hz:
        frequency = harmonic*base
        energy = window_energy(frequencies, fused, frequency)
        options.append(frequency); scores.append(energy*(2.5 if harmonic == 1 else 1.0))
        harmonic += 1
    return float(options[int(np.argmax(scores))])


def spectrum_metrics(frequencies, spectrum, target):
    cardiac = (frequencies >= .5) & (frequencies <= 3.0)
    f, power = frequencies[cardiac], spectrum[cardiac]**2
    target_energy = window_energy(f, power, target)
    possible = f[np.abs(f-target) > .3]
    distractor = max((window_energy(f, power, center) for center in possible), default=0.)
    probability = power/(np.sum(power)+1e-12)
    entropy = float(-np.sum(probability*np.log(probability+1e-12))/np.log(len(power)))
    preferred = float(f[np.argmax(power)])
    return target_energy/(distractor+1e-12), entropy, preferred


def analyze_signal(raw, ecg_hr, fs=100):
    band_signals = np.stack([BandPassFilter(raw, low, high, 4, fs) for low, high in HEART_V6_BANDS])
    rectified = np.abs(band_signals)
    full_spectra = np.abs(np.fft.fft(rectified, axis=-1))
    full_frequencies = np.fft.fftfreq(len(raw), 1/fs)
    positive = (full_frequencies > 0) & (full_frequencies <= 10)
    frequencies, spectra = full_frequencies[positive], full_spectra[:, positive]
    fused = np.sum(spectra, axis=0)
    candidates, totals, term_counts, current_scores, normalized_scores = candidate_scores(frequencies, fused)
    current_base = float(candidates[np.argmax(current_scores)])
    normalized_base = float(candidates[np.argmax(normalized_scores)])
    current_selected = select_from_family(current_base, frequencies, fused)
    normalized_selected = select_from_family(normalized_base, frequencies, fused)
    domain = (frequencies > .7) & (frequencies <= 3.0)
    argmax_selected = float(frequencies[domain][np.argmax(fused[domain])])
    narrow = BandPassFilter(raw, current_selected-.15, current_selected+.15, 1, fs)
    narrow = BandPassFilter(narrow, current_selected-.15, current_selected+.15, 2, fs)
    peaks, _ = find_peaks(narrow, height=0)
    intervals = np.diff(peaks)/fs
    final_hr = float(round(60/np.mean(intervals))) if len(intervals) else np.nan
    interval_cv = float(np.std(intervals)/(np.mean(intervals)+1e-12)) if len(intervals) else np.nan
    order = np.argsort(current_scores)[::-1]
    margin = float((current_scores[order[0]]-current_scores[order[1]])/(current_scores[order[0]]+1e-12))
    target = ecg_hr/60
    band_metrics = [spectrum_metrics(frequencies, spectrum, target) for spectrum in spectra]
    ratios, entropies, preferences = map(np.asarray, zip(*band_metrics))
    rounded = np.round(preferences/.1)*.1
    values, counts = np.unique(rounded, return_counts=True)
    modal = float(values[np.argmax(counts)])
    cross_agreement = float(np.mean(np.abs(preferences-modal) <= .15))
    target_agreement = float(np.mean(np.abs(preferences-target) <= .15))
    return {
        "frequencies": frequencies, "fused": fused, "candidates": candidates,
        "score_total": totals, "term_count": term_counts, "current_scores": current_scores,
        "normalized_scores": normalized_scores, "current_base": current_base,
        "normalized_base": normalized_base,
        "current_selected": current_selected, "normalized_selected": normalized_selected,
        "argmax_selected": argmax_selected, "final_hr": final_hr,
        "interval_cv": interval_cv, "peak_count": len(peaks), "score_margin": margin,
        "mean_td": float(np.mean(ratios)), "min_td": float(np.min(ratios)),
        "mean_entropy": float(np.mean(entropies)), "cross_agreement": cross_agreement,
        "target_agreement": target_agreement, "modal_preferred": modal,
        "preferences": preferences, "raw": raw, "narrow": narrow, "peaks": peaks,
    }


def error_kind(estimate, reference):
    error = abs(estimate-reference)
    ratio = estimate/reference
    if error <= 10: return "not_gross"
    if .45 <= ratio <= .55: return "half_rate"
    if 1.9 <= ratio <= 2.1: return "double_rate"
    return "other_gross"


def metrics(rows, key):
    errors = np.asarray([abs(float(row[key])-float(row["ecg_hr"])) for row in rows])
    return {"n": len(errors), "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "within_3_percent": float(100*np.mean(errors <= 3)),
            "within_5_percent": float(100*np.mean(errors <= 5)),
            "within_10_percent": float(100*np.mean(errors <= 10)),
            "half_rate_percent": float(100*np.mean([error_kind(float(r[key]),float(r["ecg_hr"])) == "half_rate" for r in rows])),
            "double_rate_percent": float(100*np.mean([error_kind(float(r[key]),float(r["ecg_hr"])) == "double_rate" for r in rows])),
            "other_gross_percent": float(100*np.mean([error_kind(float(r[key]),float(r["ecg_hr"])) == "other_gross" for r in rows]))}


def summarize_groups(rows, group_key):
    output=[]
    for group in sorted({row[group_key] for row in rows}):
        selected=[row for row in rows if row[group_key] == group]
        s, f = metrics(selected,"spectral_hr"), metrics(selected,"final_hr")
        output.append({group_key:group,"n":len(selected),"spectral_mae":s["mae"],"final_mae":f["mae"],
                       "mean_refinement_effect":np.mean([float(x["refinement_effect_bpm"]) for x in selected]),
                       "worsened_percent":100*np.mean([x["refinement_class"] == "worsened" for x in selected])})
    return output


def classify_stage(row):
    if float(row["final_error_bpm"]) <= 5: return "no_material_failure"
    weak = float(row["mean_target_to_distractor_ratio"]) < 1 or float(row["ecg_target_band_agreement"]) < .5
    refinement = float(row["spectral_error_bpm"]) <= 3 and float(row["refinement_effect_bpm"]) > 2
    scoring = (not weak and float(row["spectral_error_bpm"]) > 10 and
               (abs(float(row["argmax_hr"])-float(row["ecg_hr"])) <= 5 or
                abs(float(row["normalized_score_hr"])-float(row["ecg_hr"])) <= 5 or
                abs(60*float(row["modal_preferred_frequency_hz"])-float(row["ecg_hr"])) <= 5))
    matches = sum((weak, refinement, scoring))
    if matches != 1: return "unresolved_mixed"
    return "representation_failure" if weak else "refinement_failure" if refinement else "fusion_scoring_failure"


def plot_results(rows, candidate_rows, output):
    def array(key): return np.asarray([float(row[key]) for row in rows])
    spectral_error, final_error = array("spectral_error_bpm"), array("final_error_bpm")
    effect = array("refinement_effect_bpm")
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    axes[0,0].scatter(spectral_error,final_error,s=5,alpha=.25); limit=max(np.percentile(final_error,99),30)
    axes[0,0].plot([0,limit],[0,limit],"k--"); axes[0,0].set(xlim=(0,limit),ylim=(0,limit),xlabel="Spectral error (BPM)",ylabel="Final error (BPM)")
    axes[0,1].hist(np.clip(effect,-30,30),bins=61); axes[0,1].axvline(0,color="k",linestyle="--"); axes[0,1].set(xlabel="Refinement effect (BPM)",ylabel="Windows")
    axes[1,0].scatter(array("interval_cv"),effect,s=5,alpha=.2); axes[1,0].set(xlabel="Interval CV",ylabel="Refinement effect (BPM)",ylim=(-30,30))
    axes[1,1].scatter(array("spectral_interval_disagreement_bpm"),effect,s=5,alpha=.2); axes[1,1].set(xlabel="|spectral - interval| (BPM)",ylabel="Refinement effect (BPM)",ylim=(-30,30))
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"refinement_analysis.png",dpi=180); plt.close(fig)

    fig, axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].hist(array("selected_frequency_hz"),bins=np.arange(.65,3.06,.1)); axes[0].set(xlabel="Selected frequency (Hz)",ylabel="Windows")
    sample=candidate_rows[::max(1,len(candidate_rows)//50000)]
    axes[1].scatter([float(x["candidate_frequency_hz"]) for x in sample],[float(x["current_score"]) for x in sample],s=3,alpha=.15,label="current")
    axes[1].scatter([float(x["candidate_frequency_hz"]) for x in sample],[float(x["term_normalized_score"]) for x in sample],s=3,alpha=.15,label="term normalized")
    axes[1].set(xlabel="Candidate frequency (Hz)",ylabel="Score",yscale="log"); axes[1].legend()
    axes[2].scatter([float(x["candidate_frequency_hz"]) for x in sample],[float(x["valid_harmonic_terms"]) for x in sample],s=3,alpha=.12)
    axes[2].set(xlabel="Candidate frequency (Hz)",ylabel="Valid harmonic terms")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"harmonic_boundary_analysis.png",dpi=180); plt.close(fig)


def representative_plot(case, row, output):
    time=np.arange(len(case["raw"]))/100
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(time,case["raw"],linewidth=.6); axes[0,0].set(title=row["failure_stage"],xlabel="Time (s)",ylabel="BCG z-score")
    axes[0,1].plot(case["frequencies"],case["fused"]); axes[0,1].axvline(float(row["ecg_hr"])/60,color="g",linestyle="--",label="ECG"); axes[0,1].axvline(case["current_selected"],color="r",linestyle=":",label="selected"); axes[0,1].set(xlim=(.5,3),title="Fused rectified spectrum"); axes[0,1].legend()
    axes[1,0].plot(case["candidates"],case["current_scores"],label="current"); axes[1,0].plot(case["candidates"],case["normalized_scores"],label="term-normalized"); axes[1,0].set(title="Candidate scoring",xlabel="Frequency (Hz)"); axes[1,0].legend()
    axes[1,1].plot(time,case["narrow"],linewidth=.7); axes[1,1].plot(case["peaks"]/100,case["narrow"][case["peaks"]],"rx"); axes[1,1].set(title=f"Spectral {float(row['spectral_hr']):.1f}, final {float(row['final_hr']):.1f}, ECG {float(row['ecg_hr']):.1f}",xlabel="Time (s)")
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/f"representative_{row['failure_stage']}.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True); parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    baseline={row["filename"]:row for row in read_csv(args.baseline)}
    manifest=read_csv(args.manifest); rows=[]; candidate_rows=[]
    for index, provenance in enumerate(manifest):
        name=provenance["source_filename"]
        if name not in baseline: continue
        pair=np.load(args.data_dir/name)
        raw=zscore_normalize(BandPassFilter(pair[0],.5,25,4,100,padlen=500))
        ecg=float(baseline[name]["ecg_hr"]); result=analyze_signal(raw,ecg)
        if not np.isfinite(result["final_hr"]): continue
        spectral_hr=60*result["current_selected"]; final_hr=result["final_hr"]
        spectral_error=abs(spectral_hr-ecg); final_error=abs(final_hr-ecg)
        effect=final_error-spectral_error; delta=final_hr-spectral_hr
        row={"source_filename":name,"continuous_source_id":provenance["continuous_source_id"],
             "legacy_dataset_id":provenance["legacy_dataset_id"],"ecg_hr":ecg,
             "selected_frequency_hz":result["current_selected"],"spectral_hr":spectral_hr,"final_hr":final_hr,
             "current_base_frequency_hz":result["current_base"],
             "normalized_base_frequency_hz":result["normalized_base"],
             "spectral_error_bpm":spectral_error,"final_error_bpm":final_error,
             "signed_refinement_delta_bpm":delta,"absolute_refinement_delta_bpm":abs(delta),
             "refinement_effect_bpm":effect,"refinement_class":"improved" if effect<-.5 else "worsened" if effect>.5 else "unchanged",
             "interval_cv":result["interval_cv"],"peak_count":result["peak_count"],"score_margin":result["score_margin"],
             "mean_target_to_distractor_ratio":result["mean_td"],"min_target_to_distractor_ratio":result["min_td"],
             "mean_spectral_entropy":result["mean_entropy"],"cross_band_agreement":result["cross_agreement"],
             "ecg_target_band_agreement":result["target_agreement"],"modal_preferred_frequency_hz":result["modal_preferred"],
             "boundary_proximity_hz":min(result["current_selected"]-.7,3-result["current_selected"]),
             "spectral_interval_disagreement_bpm":abs(delta),"argmax_hr":60*result["argmax_selected"],
             "normalized_score_hr":60*result["normalized_selected"]}
        row["failure_stage"]=classify_stage(row); rows.append(row)
        fused_energy=float(np.sum(result["fused"]))
        for f,total,count,current,normalized in zip(result["candidates"],result["score_total"],result["term_count"],result["current_scores"],result["normalized_scores"]):
            candidate_rows.append({"source_filename":name,"candidate_frequency_hz":f,"valid_harmonic_terms":int(count),
                                   "total_score":total,"current_score":current,"term_normalized_score":normalized,
                                   "fused_energy_normalized_total":total/(fused_energy+1e-12)})
        if (index+1)%500==0: print(f"processed {index+1}/{len(manifest)}",flush=True)
    write_csv(args.output_dir/"per_window.csv",rows); write_csv(args.output_dir/"candidate_scores.csv",candidate_rows)
    write_csv(args.output_dir/"by_continuous_source.csv",summarize_groups(rows,"continuous_source_id"))
    write_csv(args.output_dir/"by_legacy_dataset_id.csv",summarize_groups(rows,"legacy_dataset_id"))
    severity=[]
    for label,low,high in (("0-3",0,3),("3-5",3,5),("5-10",5,10),(">10",10,np.inf)):
        selected=[r for r in rows if float(r["spectral_error_bpm"])>low and float(r["spectral_error_bpm"])<=high] if low else [r for r in rows if float(r["spectral_error_bpm"])<=high]
        if selected: severity.append({"spectral_error_stratum":label,**metrics(selected,"spectral_hr"),"final_mae":metrics(selected,"final_hr")["mae"],"mean_refinement_effect":np.mean([float(r["refinement_effect_bpm"]) for r in selected]),"worsened_percent":100*np.mean([r["refinement_class"]=="worsened" for r in selected])})
    write_csv(args.output_dir/"by_spectral_error_severity.csv",severity)
    diagnostics={}
    for key in ("interval_cv","peak_count","spectral_interval_disagreement_bpm","mean_spectral_entropy","mean_target_to_distractor_ratio"):
        rho,p=spearmanr([float(r[key]) for r in rows],[float(r["refinement_effect_bpm"]) for r in rows])
        diagnostics[key]={"spearman_rho":float(rho),"p_value_descriptive_only":float(p)}
    methods={name:metrics(rows,key) for name,key in (("current_harmonic","spectral_hr"),("simple_argmax","argmax_hr"),("term_normalized","normalized_score_hr"),("final_heart_v6","final_hr"))}
    for name,key in (("current_harmonic","spectral_hr"),("simple_argmax","argmax_hr"),("term_normalized","normalized_score_hr")):
        methods[name]["lower_boundary_percent"]=100*np.mean([abs(float(r[key])/60-.7)<=.05 for r in rows])
        methods[name]["upper_boundary_percent"]=100*np.mean([float(r[key])/60>=2.85 for r in rows])
    stages={stage:{"n":sum(r["failure_stage"]==stage for r in rows),"percent_all":100*np.mean([r["failure_stage"]==stage for r in rows])} for stage in sorted({r["failure_stage"] for r in rows})}
    result={"dataset_interpretation":"single_person_repeated_recordings","n":len(rows),"methods":methods,
            "refinement_classes":{label:{"n":sum(r["refinement_class"]==label for r in rows),"percent":100*np.mean([r["refinement_class"]==label for r in rows])} for label in ("improved","worsened","unchanged")},
            "diagnostic_associations":diagnostics,"failure_stages":stages,
            "taxonomy":{"material_failure":"final absolute error > 5 BPM","representation_failure":"material failure AND (mean T/D < 1 OR ECG-target band agreement < 0.5), exclusively","refinement_failure":"material failure AND spectral error <= 3 BPM AND refinement effect > 2 BPM, exclusively","fusion_scoring_failure":"material failure, adequate representation, spectral error > 10 BPM, and argmax/term-normalized/modal-band candidate within 5 BPM, exclusively","unresolved_mixed":"zero or multiple mechanism definitions satisfied"}}
    (args.output_dir/"summary.json").write_text(json.dumps(result,indent=2))
    plot_results(rows,candidate_rows,args.output_dir)
    for stage in ("representation_failure","fusion_scoring_failure","refinement_failure","unresolved_mixed"):
        pool=[r for r in rows if r["failure_stage"]==stage]
        if not pool: continue
        chosen=max(pool,key=lambda r:float(r["final_error_bpm"]))
        pair=np.load(args.data_dir/chosen["source_filename"]); raw=zscore_normalize(BandPassFilter(pair[0],.5,25,4,100,padlen=500))
        representative_plot(analyze_signal(raw,float(chosen["ecg_hr"])),chosen,args.output_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
