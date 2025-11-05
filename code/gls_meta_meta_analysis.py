
CONFIG = {
    "tau2_method": "PM",  # 可选: "DL" | "PM" | "REML"
    "use_hartung_knapp": True,
    "force_fixed_when_homogeneous": False,  
    "min_studies_for_small_study_tests": 10,
    "prefer_unified_timepoint": False,  
    "prefer_latest": True,            
    "prefer_max_coverage": True,      
    "or_to_rr_baseline_risk": None,   
    # 权重分配配置
    "use_enhanced_weights": True,    
    "weight_sample_size_power": 0.5,  
    "weight_quality_multiplier": True, 
    "show_weight_info": True,        
    "min_se_adjustment": 0.02,        
    "max_weight_ratio": 10.0,         
    # CI最小跨幅（按效应点估计的比例）
    "min_ci_width_ratio_rr": 0.10,    
    "min_ci_width_ratio_md": 0.08,    
}

import sys, platform
import pandas as pd, numpy as np
import scipy, matplotlib
import os
import matplotlib.pyplot as plt
print(f"Python: {sys.version.split()[0]}, OS: {platform.platform()}")
print(f"pandas: {pd.__version__}, numpy: {np.__version__}, scipy: {scipy.__version__}, matplotlib: {matplotlib.__version__}")

AMSTAR_ORDER = {"high": 3, "moderate": 2, "low": 1, "critically low": 0}

def normalize_amstar(x: str) -> str:
    if not isinstance(x, str): return "low"
    x = x.strip().lower()
    mapping = {
        "high": "high", "moderate": "moderate", "low": "low",
        "very low": "critically low", "critically low": "critically low"
    }
    return mapping.get(x, "low")

# 读取数据
df = pd.read_csv(r"D:\article_tho\copd_inhalation_tcm\TEST\META\LIT\scripts\data\all.csv", sep="\t")

# 列重命名
df = df.rename(columns={
    "Phase":"phase",
    "P_value":"P-value",
    "model_reported":"Model",
    "effect_type":"Effect Size"
})

# 值归一化
df["phase"] = df["phase"].astype(str).str.strip().replace({
    "Post-operation":"post-extubation",
    "Intra-operation":"intraoperative",
    "3Phases":"mixed"
}).str.lower()
df["Model"] = df["Model"].astype(str).str.strip().replace({
    "REM":"Random","FIXED":"Fixed"
})
df["Effect Size"] = df["Effect Size"].astype(str).str.strip().str.lower().replace({"wmd":"md"})
df["AMSTAR_level"] = df["AMSTAR_level"].astype(str).str.strip().str.title().replace({"Critically Low":"Critically low"})
# 规范一列供筛选
_df_amstar_norm = df["AMSTAR_level"].map(normalize_amstar)
df["amstar_norm"] = _df_amstar_norm

# 解析CI -> effect_size/ci_lower/ci_upper/se
import re
def parse_ci(s):
    if pd.isna(s): return [np.nan, np.nan, np.nan]
    m = re.search(r'([-\d.]+)\s*\(([-\d.]+)\s*,\s*([-\d.]+)\)', str(s).replace(' ', ''))
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else [np.nan,np.nan,np.nan]
df[["effect_size","ci_lower","ci_upper"]] = df["Value(95% CI)"].apply(parse_ci).tolist()
from scipy import stats
df["se"] = (df["ci_upper"] - df["ci_lower"]) / (2 * stats.norm.ppf(0.975))

# 清洗 P-value（转数值）
df["P-value"] = (df["P-value"].astype(str).str.replace("＜","<").str.replace("＞",">").str.replace(" ", ""))
def parse_pv(x):
    if isinstance(x,str) and x.startswith("<"):
        try: return float(x[1:])
        except: return np.nan
    try: return float(x)
    except: return np.nan
df["P-value"] = df["P-value"].apply(parse_pv)

def prefer_unified_phase_time(df: pd.DataFrame) -> pd.DataFrame:
    # 对每个Outcome内，优先出现次数最多的 phase 或预设优先级
    priority = ["preoxygenation", "intraoperative", "post-extubation", "mixed", "unknown"]
    if "phase" not in df.columns: return df
    df = df.copy()
    phase_counts = df.groupby(["Outcomes","phase"]).size().reset_index(name="n")
    # 为每个Outcome选一个最佳phase（先看频次，再按优先级顺序）
    phase_order = {p: i for i, p in enumerate(priority)}
    phase_counts["phase_rank"] = phase_counts["phase"].map(phase_order).fillna(len(priority))
    best_phase = (
        phase_counts.sort_values(["Outcomes","n","phase_rank"], ascending=[True,False,True])
        .drop_duplicates("Outcomes")[ ["Outcomes","phase"] ]
        .set_index("Outcomes")["phase"].to_dict()
    )
    df["_selected_phase"] = df["Outcomes"].map(best_phase)
    return df[df["phase"] == df["_selected_phase"]]

def or_to_rr(or_value: float, p0: float) -> float:
    # RR = OR / (1 - p0 + p0*OR)
    return or_value / (1.0 - p0 + p0 * or_value)

def calculate_enhanced_weights(df: pd.DataFrame, base_weights: np.ndarray) -> np.ndarray:
    """
    计算增强权重：总权重 = 基础权重 × 质量调整因子 × 样本量调整因子
    
    Args:
        df: 包含NumPrimaryStudies和AMSTAR_level的数据框
        base_weights: 基础权重（通常是1/SE^2）
    
    Returns:
        增强后的权重数组
    """
    if not CONFIG.get("use_enhanced_weights", True):
        return base_weights
    
    # 获取原始研究数量
    num_studies = df["NumPrimaryStudies"].astype(float).fillna(1.0).values
    
    # 样本量调整因子（使用平方根避免过度加权）
    power = CONFIG.get("weight_sample_size_power", 0.5)
    sample_size_factor = np.power(num_studies, power)
    
    # 质量调整因子（基于AMSTAR等级）
    if CONFIG.get("weight_quality_multiplier", True):
        amstar_scores = {
            "high": 1.0,
            "moderate": 0.8, 
            "low": 0.6,
            "critically low": 0.4
        }
        quality_factor = df["amstar_norm"].map(amstar_scores).fillna(0.6).values
    else:
        quality_factor = np.ones(len(df))
    
    # 计算增强权重
    enhanced_weights = base_weights * sample_size_factor * quality_factor
    
    # 保守的权重限制逻辑：避免权重过度集中
    max_ratio = 0.35  
    total_weight = np.sum(enhanced_weights)
    max_single_weight = total_weight * max_ratio
    
  
    over_mask = enhanced_weights > max_single_weight
    if np.any(over_mask):
        over_total = np.sum(enhanced_weights[over_mask] - max_single_weight)
        enhanced_weights[over_mask] = max_single_weight
        if np.sum(~over_mask) > 0:
            remaining_weights = enhanced_weights[~over_mask]
            remaining_total = np.sum(remaining_weights)
            if remaining_total > 0:
                enhanced_weights[~over_mask] += over_total * (remaining_weights / remaining_total)
    
    # 确保所有研究都有合理的最小权重（避免权重为0）
    min_weight_ratio = 0.05  
    min_single_weight = total_weight * min_weight_ratio
    enhanced_weights = np.maximum(enhanced_weights, min_single_weight)
    
    # 归一化权重（保持总权重不变）
    if np.sum(enhanced_weights) > 0:
        enhanced_weights = enhanced_weights * (np.sum(base_weights) / np.sum(enhanced_weights))
    
    # 显示权重分配信息（如果启用）
    if CONFIG.get("show_weight_info", True) and True:  
        print(f"\n权重分配详情 (Outcome: {df['Outcomes'].iloc[0] if 'Outcomes' in df.columns else 'Unknown'}):")
        total_enhanced = np.sum(enhanced_weights)
        for i, (author, year, num_studies, amstar, base_w, sample_f, quality_f, enhanced_w) in enumerate(zip(
            df.get("First Author", ["Unknown"] * len(df)),
            df.get("Year", ["Unknown"] * len(df)),
            num_studies,
            df.get("amstar_norm", ["unknown"] * len(df)),
            base_weights,
            sample_size_factor,
            quality_factor,
            enhanced_weights
        )):
            weight_pct = (enhanced_w / total_enhanced * 100) if total_enhanced > 0 else 0
            print(f"  {author} ({year}): 原始研究数={num_studies:.0f}, AMSTAR={amstar}")
            print(f"    基础权重={base_w:.3f}, 样本量因子={sample_f:.3f}, 质量因子={quality_f:.3f}, 最终权重={enhanced_w:.3f}, 权重占比={weight_pct:.1f}%")
    
    return enhanced_weights

def convert_or_to_rr_if_needed(row: pd.Series, p0_global: float | None) -> pd.Series:
    if isinstance(row.get("Effect Size"), str) and row["Effect Size"].strip().lower() == "or":
        p0 = p0_global if p0_global is not None else row.get("baseline_risk", np.nan)
        if pd.notna(p0):
            row["effect_size"] = or_to_rr(row["effect_size"], p0)
            row["ci_lower"] = or_to_rr(row["ci_lower"], p0)
            row["ci_upper"] = or_to_rr(row["ci_upper"], p0)
            row["Effect Size"] = "rr"
    return row

#  OR→RR 转换
if CONFIG["or_to_rr_baseline_risk"] is not None or "baseline_risk" in df.columns:
    df = df.apply(lambda r: convert_or_to_rr_if_needed(r, CONFIG["or_to_rr_baseline_risk"]), axis=1)


if CONFIG["prefer_unified_timepoint"] and {"phase"}.issubset(set(df.columns)):
    df = prefer_unified_phase_time(df)


def _valid_ci_row(row):
    return pd.notna(row.get("ci_lower")) and pd.notna(row.get("ci_upper")) and pd.notna(row.get("effect_size"))


df_valid = df[df.apply(_valid_ci_row, axis=1)].copy()

print(f"有效数据行数: {len(df_valid)}")
print(f"包含的结局: {df_valid['Outcomes'].unique()}")

# =========================
# GLS分析部分
# =========================
try:
    import numpy.linalg as LA

    def load_cca_matrix(path_csv: str) -> pd.DataFrame:
        cca = pd.read_csv(path_csv)
        cca = cca.set_index(cca.columns[0])
        return (cca > 0).astype(int)

    def build_corr_from_cca(cca_df: pd.DataFrame, sr_ids, mode: str = "union") -> np.ndarray:
        cols = [c for c in sr_ids if c in cca_df.columns]
        if len(cols) < 2:
            return None
        sub = cca_df.loc[:, cols]
        sub_c = sub.T.groupby(level=0).max().T
        cols_unique = list(sub_c.columns)
        n = sub_c.sum(axis=0).astype(float)
        k = len(cols_unique)
        R = np.eye(k)
        for i in range(k):
            for j in range(i+1, k):
                si, sj = cols_unique[i], cols_unique[j]
                shared = (sub_c[si] & sub_c[sj]).sum()
                if mode == "union":
                    union = (sub_c[si] | sub_c[sj]).sum()
                    denom = max(1.0, float(union))  
                else:
                    ni = float(n.loc[si])
                    nj = float(n.loc[sj])
                    denom = max(1.0, min(ni, nj))
                rho = float(shared / denom)
                if i != j:
                    rho = min(rho, 0.99)
                R[i, j] = R[j, i] = rho
        return R

    def gls_pool(effects: np.ndarray, se: np.ndarray, R: np.ndarray):
        if R is None or R.shape[0] != len(effects):
            return None
        Dsqrt = np.diag(se)
        Sigma = Dsqrt @ R @ Dsqrt
        eps = 1e-10
        try:
            W = LA.inv(Sigma + eps*np.eye(Sigma.shape[0]))
        except LA.LinAlgError:
            try:
                U, s, Vh = np.linalg.svd(Sigma)
                s_inv = np.where(s > 1e-10, 1.0/s, 0.0)
                W = Vh.T @ np.diag(s_inv) @ U.T
            except LA.LinAlgError:
                return None
        one = np.ones((len(effects), 1))
        y = effects.reshape(-1,1)
        mu_den = (one.T @ W @ one).item()
        if mu_den <= 0:
            return None
        mu_num = (one.T @ W @ y).item()
        mu = mu_num / mu_den
        se_mu = (np.sqrt(1.0 / mu_den))
        # GLS weights为每个review的权重：a = W 1 / (1' W 1)
        a = (W @ one / mu_den).reshape(-1)
        if a.size > 0:
            a_sum = float(np.sum(np.abs(a)))
            if a_sum > 0:
                a_norm = a / a_sum
                max_share = 0.35  
                min_share = 0.02  
                a_clipped = np.clip(a_norm, min_share, max_share)
                a = a_clipped / float(np.sum(a_clipped))
                a = a.reshape(-1)
        resid = y - mu*one
        Q = float((resid.T @ W @ resid).item())
        k = len(effects)
        I2 = float(max(0.0, (Q - (k - 1)) / Q)) if (k > 1 and Q > 0) else 0.0
        return mu, se_mu, a, I2, Q

    # 加载CCA矩阵
    cca_path = r"D:\article_tho\copd_inhalation_tcm\TEST\META\LIT\scripts\data\CCA.csv"
    cca_df = load_cca_matrix(cca_path)
    
    # 创建输出目录
    output_dir = "data_clean_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 对所有结局进行GLS分析
    gls_rows = []
    for outcome, g in df_valid.groupby("Outcomes"):
        gg = g.dropna(subset=["effect_size","ci_lower","ci_upper"]).copy()
        if len(gg) < 3:
            print(f"跳过 {outcome}: 研究数量不足 (n={len(gg)})")
            continue
        
        # 统一效应量类型
        es_mode = gg["Effect Size"].str.lower().mode()
        if es_mode.empty:
            print(f"跳过 {outcome}: 无法确定效应量类型")
            continue
        es_type = es_mode.iat[0]
        gg = gg[gg["Effect Size"].str.lower() == es_type]
        if len(gg) < 3:
            print(f"跳过 {outcome}: 统一效应量类型后研究数量不足 (n={len(gg)})")
            continue
        
        # 先基于SE去重：同一SR_ID仅保留标准误更小的那条
        if es_type in ["rr","or"]:
            tmp_se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
        else:
            tmp_se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
        gg = gg.copy()
        # 保留原始顺序，同时为每个 SR_ID 选择最小SE那条记录
        gg = gg.reset_index(drop=False).rename(columns={"index": "_orig_idx_"})
        gg["_tmp_se_"] = tmp_se
        best_idx = gg.groupby("SR_ID")["_tmp_se_"].idxmin()
        gg = gg.loc[best_idx].copy()
        gg = gg.sort_values("_orig_idx_")
        
        print(f"\n处理结局: {outcome} (效应量类型: {es_type}, 研究数: {len(gg)})")
        
        # 构造 y 与 se（去重后）
        if es_type in ["rr","or"]:
            y = np.log(gg["effect_size"].astype(float).values)
            se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
        else:
            y = gg["effect_size"].astype(float).values
            se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
        
        # 应用最小标准误安全限制
        min_se = CONFIG.get("min_se_adjustment", 0.01)
        se = np.maximum(se, min_se)
        
        sr_ids = gg["SR_ID"].astype(str).tolist()
        R = build_corr_from_cca(cca_df, sr_ids, mode="union")
        
        # 输出相关性矩阵详情
        if R is not None:
            print(f"\n结局 {outcome} 的相关性矩阵R:")
            print(R.round(3))  # 保留3位小数，便于观察
            # 检查是否有异常高的相关性
            off_diag = R[np.triu_indices_from(R, k=1)]
            if len(off_diag) > 0:
                max_corr = np.max(off_diag)
                mean_corr = np.mean(off_diag)
                print(f"  非对角线相关系数: 最大值={max_corr:.3f}, 平均值={mean_corr:.3f}")
                if max_corr > 0.8:
                    print(f"  警告: 存在异常高的相关性 (>0.800)，可能影响权重分配")
        else:
            print(f"  警告: 无法构建相关性矩阵，使用独立分析")
            R = np.eye(len(y))
        
        # 对GLS分析应用增强权重（通过调整标准误）
        if CONFIG.get("use_enhanced_weights", True):
            base_weights = 1.0 / (se**2)
            enhanced_weights = calculate_enhanced_weights(gg, base_weights)
            
            weight_pcts = (enhanced_weights / np.sum(enhanced_weights) * 100)
            zero_weight_count = np.sum(weight_pcts < 0.1)  
            if zero_weight_count > 0:
                print(f"  警告: {zero_weight_count}个研究权重过低 (<0.1%)，可能导致分析偏差")
            
            se_adjusted = se * np.sqrt(base_weights / enhanced_weights)
            se_adjusted = np.maximum(se_adjusted, min_se)
            print(f"  结局 {outcome} 的se_adjusted: {se_adjusted.round(4)}")
            res = gls_pool(y, se_adjusted, R)
        else:
            res = gls_pool(y, se, R)
        
        if res is None:
            print(f"  错误: GLS分析失败")
            continue
        
        mu, se_mu, weights, I2, Q = res
        # 输出se_mu调整前后的值
        se_mu_original = se_mu
        se_mu = max(se_mu, min_se)
        print(f"  结局 {outcome} 的se_mu（调整前）: {se_mu_original:.6f}, 调整后: {se_mu:.6f}")
        # Hartung–Knapp 风格的方差膨胀：用Q/(k-1)缩放协方差（不小于1）
        k_eff = len(gg)
        if k_eff > 1:
            hk_c = max(1.0, Q / max(1.0, (k_eff - 1)))
        else:
            hk_c = 1.0
        se_mu_hk = float(np.sqrt(hk_c) * se_mu)
        # 使用t分布分位数
        df_hk = max(1, k_eff - 1)
        tcrit = float(stats.t.ppf(0.975, df_hk))
        lo, hi = float(mu - tcrit*se_mu_hk), float(mu + tcrit*se_mu_hk)
        
        if es_type in ["rr","or"]:
            mu_d, lo_d, hi_d = float(np.exp(mu)), float(np.exp(lo)), float(np.exp(hi))
        else:
            mu_d, lo_d, hi_d = float(mu), float(lo), float(hi)
        
        # 强化标准误的安全约束：增加总体效应CI的最小间隔限制
        if es_type in ["rr","or"]:
            min_ratio = CONFIG.get("min_ci_width_ratio_rr", 0.10)
            min_ci_gap = abs(mu_d) * float(min_ratio)
        else:
            min_ratio = CONFIG.get("min_ci_width_ratio_md", 0.08)
            min_ci_gap = abs(mu_d) * float(min_ratio)
        actual_gap = hi_d - lo_d
        if actual_gap < min_ci_gap:
            mid = (lo_d + hi_d) / 2
            lo_d = mid - min_ci_gap / 2
            hi_d = mid + min_ci_gap / 2
            print(f"  结局 {outcome} 的CI间隔过小 ({actual_gap:.6f} < {min_ci_gap:.6f})，已调整为 {lo_d:.3f}-{hi_d:.3f}")
        
        gls_rows.append({
            "Outcome": outcome,
            "Effect Type": es_type,
            "k_reviews": len(gg),
            "GLS Effect": mu_d,
            "GLS CI Lower": lo_d,
            "GLS CI Upper": hi_d,
            "Corr mode": "overlap/min(n_i,n_j)",
            "I2_GLS": I2,
            "Q_GLS": Q
        })
        
        print(f"  结果: {es_type.upper()}={mu_d:.3f} ({lo_d:.3f} to {hi_d:.3f}), I²={I2*100:.1f}%")

    # 保存GLS结果
    if gls_rows:
        gls_df = pd.DataFrame(gls_rows)
        output_file = os.path.join(output_dir, "gls_analysis_no_amstar.xlsx")
        with pd.ExcelWriter(output_file) as writer:
            gls_df.to_excel(writer, sheet_name="GLS_Results", index=False)
        print(f"\nGLS分析结果已保存到: {output_file}")
        print(f"共分析了 {len(gls_rows)} 个结局")
        
        # 显示结果摘要
        print("\n=== GLS分析结果摘要 ===")
        print(gls_df[["Outcome", "Effect Type", "k_reviews", "GLS Effect", "GLS CI Lower", "GLS CI Upper", "I2_GLS"]].to_string(index=False))
    else:
        print("没有可分析的结局")

    # =========================
    # 绘制森林图
    # =========================
    def plot_gls_forest_plots(df_layer: pd.DataFrame):
        """绘制GLS森林图，保持与原始脚本相同的样式"""
        fig_dir = os.path.join(output_dir, "fig_gls_all")
        os.makedirs(fig_dir, exist_ok=True)
        
        for outcome, g in df_layer.groupby("Outcomes"):
            gg = g.dropna(subset=["effect_size","ci_lower","ci_upper"]).copy()
            if len(gg) < 3:
                continue
            
            # 统一效应量类型
            es_mode = gg["Effect Size"].str.lower().mode()
            if es_mode.empty:
                continue
            es_type = es_mode.iat[0]
            gg = gg[gg["Effect Size"].str.lower() == es_type]
            if len(gg) < 3:
                continue
            
            print(f"绘制森林图: {outcome}")
            
            # 基于SE去重：同一SR_ID仅保留标准误更小的那条
            if es_type in ["rr","or"]:
                tmp_se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
            else:
                tmp_se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
            gg = gg.copy()
            # 保留原始顺序，同时为每个 SR_ID 选择最小SE那条记录
            gg = gg.reset_index(drop=False).rename(columns={"index": "_orig_idx_"})
            gg["_tmp_se_"] = tmp_se
            best_idx = gg.groupby("SR_ID")["_tmp_se_"].idxmin()
            gg = gg.loc[best_idx].copy()
            gg = gg.sort_values("_orig_idx_")
            
            # 构造 y 与 se（去重后）
            if es_type in ["rr","or"]:
                y = np.log(gg["effect_size"].astype(float).values)
                se = (np.log(gg["ci_upper"].astype(float).values) - np.log(gg["ci_lower"].astype(float).values)) / (2 * stats.norm.ppf(0.975))
                x_vals = gg["effect_size"].astype(float).values
                x_lo = gg["ci_lower"].astype(float).values
                x_hi = gg["ci_upper"].astype(float).values
                x_label = f"{es_type.upper()} (ratio)"
            else:
                y = gg["effect_size"].astype(float).values
                se = ((gg["ci_upper"].astype(float) - gg["ci_lower"].astype(float)) / (2 * stats.norm.ppf(0.975))).values
                x_vals = gg["effect_size"].astype(float).values
                x_lo = gg["ci_lower"].astype(float).values
                x_hi = gg["ci_upper"].astype(float).values
                x_label = "Mean Difference"
            
            # 应用最小标准误安全限制
            min_se = CONFIG.get("min_se_adjustment", 0.01)
            se = np.maximum(se, min_se)
            
            # 构建相关性矩阵
            sr_ids = gg["SR_ID"].astype(str).tolist()
            R = build_corr_from_cca(cca_df, sr_ids, mode="union")
            if R is None:
                R = np.eye(len(y))
            
            # 应用增强权重
            if CONFIG.get("use_enhanced_weights", True):
                base_weights = 1.0 / (se**2)
                enhanced_weights = calculate_enhanced_weights(gg, base_weights)
                se_adjusted = se * np.sqrt(base_weights / enhanced_weights)
                # 再次应用最小标准误限制，防止过度调整
                se_adjusted = np.maximum(se_adjusted, min_se)
                res = gls_pool(y, se_adjusted, R)
            else:
                res = gls_pool(y, se, R)
            
            if res is None:
                continue
            
            mu, se_mu, weights, I2, Q = res
            # 应用最小SE限制，防止计算后SE过小
            se_mu = max(se_mu, min_se)
            # Hartung–Knapp 风格膨胀与t分位数
            k_eff = len(gg)
            if k_eff > 1:
                hk_c = max(1.0, Q / max(1.0, (k_eff - 1)))
            else:
                hk_c = 1.0
            se_mu_hk = float(np.sqrt(hk_c) * se_mu)
            df_hk = max(1, k_eff - 1)
            tcrit = float(stats.t.ppf(0.975, df_hk))
            lo, hi = float(mu - tcrit*se_mu_hk), float(mu + tcrit*se_mu_hk)
            
            # 计算展示值
            if es_type in ["rr","or"]:
                disp = np.exp(y)
                mu_d, lo_d, hi_d = float(np.exp(mu)), float(np.exp(lo)), float(np.exp(hi))
                x_overall = (mu_d, lo_d, hi_d)
            else:
                disp = y
                mu_d, lo_d, hi_d = float(mu), float(lo), float(hi)
                x_overall = (mu_d, lo_d, hi_d)
            
            # 绘制森林图
            try:
                labels = (
                    gg["First Author"].astype(str)
                    + " (" + gg["Year"].astype(str) + ")"
                ).tolist()
                
                # 计算权重百分比（直接使用正则化后的GLS权重）
                w = np.abs(weights)
                wsum = float(np.nansum(w))
                if wsum == 0 or not np.isfinite(wsum):
                    weight_pct = np.full_like(w, 100.0/len(w), dtype=float)
                else:
                    weight_pct = (w / wsum) * 100.0
                # 限制显示的最大/最小百分比，避免0%或极端偏高
                weight_pct = np.clip(weight_pct, 2.0, 35.0)
                # 归一化到总和为100
                weight_pct = weight_pct * (100.0 / float(np.sum(weight_pct)))
                
                # 限制权重百分比用于绘图
                weight_pct_plot = np.clip(np.nan_to_num(weight_pct, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
                left_labels = [f"{lab}  [w={w:.1f}%]" for lab, w in zip(labels, weight_pct)]
                right_texts = [f"{m:.2g} ({loi:.2g} to {hii:.2g})" for m, loi, hii in zip(x_vals, x_lo, x_hi)]
                
                y_pos = np.arange(len(labels))[::-1]
                plt.figure(figsize=(9, 0.5*len(labels) + 2.8))
                ax = plt.gca()
                
                # 绘制单个研究的CI线和点
                for i, (m, lo_i, hi_i, w) in enumerate(zip(x_vals, x_lo, x_hi, weight_pct_plot)):
                    ax.plot([lo_i, hi_i], [y_pos[i], y_pos[i]], color="#6c757d", lw=2)
                    ax.scatter(m, y_pos[i], s=max(10.0, 30.0 + 2.0*w), color="#2c7fb8", zorder=3)
                
                # 绘制总体效应（菱形）
                ax.plot([x_overall[1], x_overall[2]], [-1, -1], color="#d62728", linewidth=3)
                ax.scatter(x_overall[0], -1, s=80, marker="D", color="#d62728", zorder=4)
                
                # 设置y轴标签
                yticks = list(y_pos) + [-1]
                ylabels = left_labels + ["Overall (GLS)"]
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                ax.set_xlabel(x_label)
                
                # 右侧文本（效应量和CI）
                for i, txt in enumerate(right_texts):
                    ax.text(ax.get_xlim()[1], y_pos[i], "  " + txt, va="center", ha="left", fontsize=10)
                
                # 总体效应文本
                if es_type in ["rr","or"]:
                    overall_txt = f"RR={x_overall[0]:.2g} ({x_overall[1]:.2g} to {x_overall[2]:.2g})"
                else:
                    overall_txt = f"{x_overall[0]:.2g} ({x_overall[1]:.2g} to {x_overall[2]:.2g})"
                ax.text(ax.get_xlim()[1], -1, "  " + overall_txt, va="center", ha="left", fontsize=10, fontweight="bold")
                
                # 标题
                ax.set_title(
                    f"{outcome} — GLS meta-meta analysis\n"
                    f"Model: GLS Common-Effect, k={len(labels)}, I²≈{I2*100:.1f}%",
                    fontsize=12
                )
                ax.grid(axis="x", linestyle=":", alpha=0.35)
                plt.tight_layout()
                
                # 保存图片
                safe_outcome = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(outcome))
                fpath = os.path.join(fig_dir, f"forest_gls_{safe_outcome}_{es_type}.png")
                plt.savefig(fpath, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"  森林图已保存: {fpath}")
                
            except Exception as e:
                print(f"  绘制森林图失败: {e}")
                continue

    # 运行森林图绘制
    if gls_rows:
        print("\n开始绘制森林图...")
        plot_gls_forest_plots(df_valid)
        print(f"森林图已保存到: {os.path.join(output_dir, 'fig_gls_all')}")

except Exception as e:
    import traceback
    print(f"GLS分析失败: {e}")
    traceback.print_exc()