# Meta–meta analysis, no GLS, no AMSTAR layers (all comparators)
import sys, os, re, platform
import pandas as pd, numpy as np
import scipy, matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as student_t


# ---------------------
# Config
# ---------------------
CONFIG = {
    "tau2_method": "PM",            # "DL" | "PM" | "REML"
    "use_hartung_knapp": True,
    "heterogeneity_i2_threshold": 0,  # (deprecated: always use Random model)
    # OR->RR conversion
    "or_to_rr_baseline_risk": None,      # e.g., 0.2; if None, use per-row baseline_risk if present
    # Enhanced weighting
    "use_enhanced_weights": False,
    "weight_sample_size_power": 0.5,
    "weight_quality_multiplier": True,
    "show_weight_info": True,
}

AMSTAR_ORDER = {"high": 3, "moderate": 2, "low": 1, "critically low": 0}

def normalize_amstar(x: str) -> str:
    if not isinstance(x, str):
        return "low"
    x = x.strip().lower()
    mapping = {
        "high": "high",
        "moderate": "moderate",
        "low": "low",
        "very low": "critically low",
        "critically low": "critically low",
    }
    return mapping.get(x, "low")

def parse_ci(s):
    if pd.isna(s):
        return [np.nan, np.nan, np.nan]
    m = re.search(r"([-\d.]+)\s*\(([-\d.]+)\s*,\s*([-\d.]+)\)", str(s).replace(" ", ""))
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else [np.nan, np.nan, np.nan]

def or_to_rr(or_value: float, p0: float) -> float:
    return or_value / (1.0 - p0 + p0 * or_value)

def convert_or_to_rr_if_needed(row: pd.Series, p0_global: float | None) -> pd.Series:
    if isinstance(row.get("Effect Size"), str) and row["Effect Size"].strip().lower() == "or":
        p0 = p0_global if p0_global is not None else row.get("baseline_risk", np.nan)
        if pd.notna(p0):
            row["effect_size"] = or_to_rr(row["effect_size"], p0)
            row["ci_lower"] = or_to_rr(row["ci_lower"], p0)
            row["ci_upper"] = or_to_rr(row["ci_upper"], p0)
            row["Effect Size"] = "rr"
    return row

def calculate_enhanced_weights(df: pd.DataFrame, base_weights: np.ndarray) -> np.ndarray:
    if not CONFIG.get("use_enhanced_weights", True):
        return base_weights
    num_studies = df.get("NumPrimaryStudies", pd.Series([1]*len(df))).astype(float).fillna(1.0).values
    power = CONFIG.get("weight_sample_size_power", 0.5)
    sample_size_factor = np.power(num_studies, power)
    if CONFIG.get("weight_quality_multiplier", True):
        amstar_scores = {"high": 1.0, "moderate": 0.8, "low": 0.6, "critically low": 0.4}
        quality_factor = df.get("amstar_norm", pd.Series(["low"]*len(df))).map(amstar_scores).fillna(0.6).values
    else:
        quality_factor = np.ones(len(df))
    enhanced_weights = base_weights * sample_size_factor * quality_factor
    if np.sum(enhanced_weights) > 0:
        enhanced_weights = enhanced_weights * (np.sum(base_weights) / np.sum(enhanced_weights))
    if CONFIG.get("show_weight_info", True) and len(df) <= 5 and len(df) >= 1:
        print(f"\n权重分配详情 (Outcome: {df['Outcomes'].iloc[0] if 'Outcomes' in df.columns else 'Unknown'}):")
        for author, year, ns, amstar, base_w, sfac, qfac, ef in zip(
            df.get("First Author", ["Unknown"]*len(df)),
            df.get("Year", ["Unknown"]*len(df)),
            num_studies,
            df.get("amstar_norm", ["unknown"]*len(df)),
            base_weights,
            sample_size_factor,
            quality_factor,
            enhanced_weights,
        ):
            print(f"  {author} ({year}): {ns:.0f} studies, AMSTAR={amstar}")
            print(f"    基础权重={base_w:.3f}, 样本量因子={sfac:.3f}, 质量因子={qfac:.3f}, 最终权重={ef:.3f}")
    return enhanced_weights

# ---------------------
# Tau estimators and pooling
# ---------------------
def tau2_dl(weights, effects):
    k = len(effects)
    w = weights
    y = effects
    ybar = np.sum(w*y)/np.sum(w)
    Q = np.sum(w*(y - ybar)**2)
    c = np.sum(w) - np.sum(w**2)/np.sum(w)
    return max(0.0, (Q - (k-1))/c) if c > 0 else 0.0

def tau2_pm(weights, effects):
    w = weights.copy()
    y = effects
    tau2 = 0.0
    for _ in range(100):
        w_star = 1.0/(1.0/w + tau2)
        ybar = np.sum(w_star*y)/np.sum(w_star)
        Q = np.sum(w_star*(y - ybar)**2)
        c = np.sum(w_star) - np.sum(w_star**2)/np.sum(w_star)
        new_tau2 = max(0.0, (Q - (len(y)-1))/c) if c > 0 else 0.0
        if abs(new_tau2 - tau2) < 1e-8:
            break
        tau2 = new_tau2
    return tau2

def tau2_reml(vi, effects):
    y = effects
    def neg_reml_loglik(tau2):
        w = 1.0/(vi + tau2)
        ybar = np.sum(w*y)/np.sum(w)
        return 0.5*( np.sum(np.log(vi + tau2)) + np.log(np.sum(w)) + np.sum(w*(y - ybar)**2) )
    grid = np.concatenate([[0.0], np.logspace(-6, 2, 80)])
    vals = [neg_reml_loglik(t) for t in grid]
    return float(grid[int(np.argmin(vals))])

def estimate_tau2(standard_errors, effects, method: str):
    vi = standard_errors**2
    weights = 1.0/vi
    m = method.upper()
    if m == "DL":
        return tau2_dl(weights, effects)
    if m == "PM":
        return tau2_pm(weights, effects)
    if m == "REML":
        return tau2_reml(vi, effects)
    return tau2_pm(weights, effects)

def pool_random(effects, se, tau2, use_hk: bool):
    vi = se**2
    wi = 1.0/(vi + tau2)
    mu = np.sum(wi*effects)/np.sum(wi)
    se_mu = np.sqrt(1.0/np.sum(wi))
    k = len(effects)
    if use_hk and k >= 3:
        y = effects
        ybar = mu
        Q = np.sum(wi*(y - ybar)**2)
        df_k = k - 1
        A = max(1.0, Q/df_k) if df_k > 0 else 1.0
        se_mu = se_mu * np.sqrt(A)
        t_crit = student_t.ppf(0.975, df_k)
        ci_lo, ci_hi = mu - t_crit*se_mu, mu + t_crit*se_mu
        return mu, se_mu, (ci_lo, ci_hi), "Random-HK"
    z = 1.96
    ci_lo, ci_hi = mu - z*se_mu, mu + z*se_mu
    return mu, se_mu, (ci_lo, ci_hi), "Random"

# ---------------------
# Forest plotting 
# ---------------------
def plot_forest_for_outcome(outcome: str, gg: pd.DataFrame, pooled_tuple, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    mu, ci_lo, ci_hi, es_type, model_used, I2 = pooled_tuple
    if es_type in ["rr", "or"]:
        x_vals = gg["effect_size"].astype(float).values
        x_lo = gg["ci_lower"].astype(float).values
        x_hi = gg["ci_upper"].astype(float).values
        overall_text = f"{es_type.upper()}={mu:.3f} ({ci_lo:.3f} to {ci_hi:.3f})"
        x_label = f"{es_type.upper()} (ratio)"
    else:
        x_vals = gg["effect_size"].astype(float).values
        x_lo = gg["ci_lower"].astype(float).values
        x_hi = gg["ci_upper"].astype(float).values
        overall_text = f"{mu:.3f} ({ci_lo:.3f} to {ci_hi:.3f})"
        x_label = "Mean Difference"

    se = gg["se"].astype(float).values
    vi = se**2
    if model_used.startswith("Random"):
        z = 1.96
        se_overall = (ci_hi - mu)/z if ci_hi > mu else (mu - ci_lo)/z
        tau2_guess = max(0.0, np.mean(vi))
        w = 1.0/(vi + tau2_guess)
    else:
        w = 1.0/vi
    wsum = float(np.nansum(w))
    if wsum == 0 or not np.isfinite(wsum):
        weight_pct = np.full_like(w, 100.0/len(w), dtype=float)
    else:
        weight_pct = (w / wsum) * 100.0
    weight_pct_plot = np.clip(np.nan_to_num(weight_pct, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    years = gg.get("Year", pd.Series([""]*len(gg)))
    years_clean = years.fillna("").astype(int).astype(str).replace("nan", "")
    labels = (gg.get("First Author", gg.index.astype(str)).astype(str) + " (" + years_clean + ")").tolist()
    right_texts = [f"{m:.3f} ({loi:.3f} to {hii:.3f})" for m, loi, hii in zip(x_vals, x_lo, x_hi)]

    y_pos = np.arange(len(labels))[::-1]
    plt.figure(figsize=(9, 0.5*len(labels) + 2.8))
    ax = plt.gca()
    for i, (m, lo_i, hi_i, w_i) in enumerate(zip(x_vals, x_lo, x_hi, weight_pct_plot)):
        ax.plot([lo_i, hi_i], [y_pos[i], y_pos[i]], color="#6c757d", lw=2)
        ax.scatter(m, y_pos[i], s=max(10.0, 30.0 + 2.0*w_i), color="#2c7fb8", zorder=3)
    ax.plot([ci_lo, ci_hi], [-1, -1], color="#d62728", linewidth=3)
    ax.scatter(mu, -1, s=80, marker="D", color="#d62728", zorder=4)

    left_labels = [f"{lab}  [w={w:.1f}%]" for lab, w in zip(labels, weight_pct)]
    yticks = list(y_pos) + [-1]
    ylabels = left_labels + ["Overall (meta-meta)"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(x_label)

    for i, txt in enumerate(right_texts):
        ax.text(ax.get_xlim()[1], y_pos[i], "  " + txt, va="center", ha="left", fontsize=10)
    ax.text(ax.get_xlim()[1], -1, "  " + overall_text, va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_title(
        f"{outcome}\n"
        f"Model: {model_used}, k={len(labels)}, I\u00b2={I2:.1f}%",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    plt.tight_layout()
    safe_outcome = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(outcome))
    fpath = os.path.join(fig_dir, f"forest_meta_meta_{safe_outcome}.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------
# Core meta–meta routine
# ---------------------
def run_meta_meta(df: pd.DataFrame, out_xlsx: str):
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    rows = []
    fig_dir = os.path.join(os.path.dirname(out_xlsx), "fig_meta_meta_all")
    for outcome, g in df.groupby("Outcomes"):
        gg = g.dropna(subset=["effect_size", "ci_lower", "ci_upper", "Effect Size"]).copy()
        if len(gg) < 3:
            continue
        es_mode = gg["Effect Size"].astype(str).str.lower().mode()
        if es_mode.empty:
            continue
        es_type = es_mode.iat[0]
        gg = gg[gg["Effect Size"].astype(str).str.lower() == es_type]
        if len(gg) < 3:
            continue

        if es_type in ["rr", "or"]:
            valid_mask = (gg["ci_upper"].astype(float) > 0) & (gg["ci_lower"].astype(float) > 0)
            gg = gg[valid_mask].copy()
            if len(gg) < 2:
                print(f"跳过 {outcome}: 有效CI数据不足 (n={len(gg)})")
                continue
            tmp_se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
        else:
            tmp_se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
        gg = gg.copy()
        gg = gg.reset_index(drop=False).rename(columns={"index": "_orig_idx_"})
        gg["_tmp_se_"] = tmp_se
        
        gg = gg.dropna(subset=["_tmp_se_"])
        if len(gg) < 2:
            print(f"跳过 {outcome}: 去除NaN后研究数量不足 (n={len(gg)})")
            continue
            
        best_idx = gg.groupby("SR_ID")["_tmp_se_"].idxmin()
        best_idx = best_idx.dropna()
        if best_idx.empty:
            print(f"跳过 {outcome}: 无法找到有效的最小SE索引")
            continue
        gg = gg.loc[best_idx].copy()
        gg = gg.sort_values("_orig_idx_")
        
        if es_type in ["rr", "or"]:
            y = np.log(gg["effect_size"].astype(float).values)
            se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
            backtransform = np.exp
            label = es_type.upper()
        else:
            y = gg["effect_size"].astype(float).values
            se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
            backtransform = lambda x: x
            label = "MD"

        vi = se**2
        wf_base = 1.0/vi
        wf = calculate_enhanced_weights(gg, wf_base)

        mu_fe = float(np.sum(wf*y)/np.sum(wf))
        se_fe = float(np.sqrt(1.0/np.sum(wf)))
        Q = float(np.sum(wf*(y - mu_fe)**2))
        k = len(y)
        I2 = float(max(0.0, (Q - (k - 1)) / Q)) * 100.0 if (k > 1 and Q > 0) else 0.0

        # Always use Random model
        tau2 = float(estimate_tau2(se, y, CONFIG.get("tau2_method", "PM")))
        mu, se_mu, (ci_lo, ci_hi), model_used = pool_random(y, se, tau2, CONFIG.get("use_hartung_knapp", True))

        disp_mu, disp_lo, disp_hi = backtransform(mu), backtransform(ci_lo), backtransform(ci_hi)
        rows.append({
            "Outcome": outcome,
            "Effect Type": label,
            "k_reviews": k,
            "Model": model_used,
            "I2": I2,
            "Pooled": float(disp_mu),
            "CI Lower": float(disp_lo),
            "CI Upper": float(disp_hi),
        })

        # Plot per-outcome forest
        plot_forest_for_outcome(
            outcome,
            gg,
            (float(disp_mu), float(disp_lo), float(disp_hi), es_type, model_used, float(I2)),
            fig_dir,
        )

    res_df = pd.DataFrame(rows)
    with pd.ExcelWriter(out_xlsx) as writer:
        if not res_df.empty:
            res_df.to_excel(writer, sheet_name="meta_meta_all", index=False)
    print("Meta-meta synthesis exported (all).")
    return res_df

def prepare_dataframe(input_path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(input_path)
    if ext.lower() in [".csv", ".tsv", ".txt"]:
        try:
            df = pd.read_csv(input_path, sep=",", encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_path, sep=",", encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, sep=",", encoding='latin-1')
        except:
            try:
                df = pd.read_csv(input_path, sep="\t", encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(input_path, sep="\t", encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(input_path, sep="\t", encoding='latin-1')
    else:
        try:
            df = pd.read_csv(input_path, sep=",", encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_path, sep=",", encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(input_path, sep=",", encoding='latin-1')
        except:
            try:
                df = pd.read_csv(input_path, sep="\t", encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(input_path, sep="\t", encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(input_path, sep="\t", encoding='latin-1')


    df = df.rename(columns={
        "Phase": "phase",
        "P_value": "P-value",
        "model_reported": "Model",
        "effect_type": "Effect Size",
    })

    df["Model"] = df.get("Model", pd.Series([np.nan]*len(df))).astype(str).str.strip().replace({"REM": "Random", "FIXED": "Fixed"})
    df["Effect Size"] = df.get("Effect Size", pd.Series([np.nan]*len(df))).astype(str).str.strip().str.lower().replace({"wmd": "md"})
    df["AMSTAR_level"] = df.get("AMSTAR_level", pd.Series(["low"]*len(df))).astype(str).str.strip().str.title().replace({"Critically Low": "Critically low"})
    df["amstar_norm"] = df["AMSTAR_level"].map(normalize_amstar)


    parsed = df.get("Value(95% CI)", pd.Series([np.nan]*len(df))).apply(parse_ci).tolist()
    if len(parsed) == len(df):
        df[["effect_size", "ci_lower", "ci_upper"]] = parsed

    if "se" not in df.columns or df["se"].isna().all():
        df["se"] = (df["ci_upper"] - df["ci_lower"]) / (2 * stats.norm.ppf(0.975))


    df["P-value"] = df.get("P-value", pd.Series([np.nan]*len(df))).astype(str).str.replace("＜", "<").str.replace("＞", ">").str.replace(" ", "")
    def parse_pv(x):
        if isinstance(x, str) and x.startswith("<"):
            try:
                return float(x[1:])
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan
    df["P-value"] = df["P-value"].apply(parse_pv)


    if CONFIG["or_to_rr_baseline_risk"] is not None or "baseline_risk" in df.columns:
        df = df.apply(lambda r: convert_or_to_rr_if_needed(r, CONFIG["or_to_rr_baseline_risk"]), axis=1)

    return df


def main():
    print(f"Python: {sys.version.split()[0]}, OS: {platform.platform()}")
    print(f"pandas: {pd.__version__}, numpy: {np.__version__}, scipy: {scipy.__version__}, matplotlib: {matplotlib.__version__}")


    if len(sys.argv) < 3:
        print("Usage: python meta_meta_subanalysis.py <input_csv_tsv> <output_xlsx>")

        default_in = r"D:\article_tho\copd_inhalation_tcm\TEST\META\LIT\scripts\data\all.csv"
        default_out = "data_clean_output/meta_all.xlsx"
        print(f"Falling back to defaults: input={default_in}, output={default_out}")
        input_path = default_in
        out_xlsx = default_out
    else:
        input_path = sys.argv[1]
        out_xlsx = sys.argv[2]

    df = prepare_dataframe(input_path)

    res_df = run_meta_meta(df, out_xlsx)

        
    if not res_df.empty:
        print("\nMeta-meta (all):")
        print(res_df[["Outcome", "Effect Type", "Pooled", "CI Lower", "CI Upper", "Model", "I2", "k_reviews"]].to_string(index=False))

if __name__ == "__main__":
    main()
