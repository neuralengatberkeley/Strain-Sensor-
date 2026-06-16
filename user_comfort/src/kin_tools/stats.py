# Stats helpers:
# - LME (random intercept only) per method (DT & TT) for APP and REM: https://www.statsmodels.org/stable/mixed_linear.html
# - Collapsed paired t-tests for time data
# - Paired tests per metric for ratings (DT vs TT)
# - Table printers for per-participant APP1/REM1 intercepts

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from scipy import stats

# More helper methods

def _sec_to_mmss_simple(sec):
    """Internal helper (unsigned) to avoid circular imports."""
    sec_i = int(round(abs(sec)))
    m, s = divmod(sec_i, 60)
    return f"{m}:{s:02d}"

def _random_intercept_value(re_series):
    """
    Extract the random-intercept value from statsmodels MixedLM random_effects entry.
    It can be keyed as 'Intercept', 0, or have no named index (then take the first value).
    """
    if re_series is None:
        return 0.0
    # dict-like (older statsmodels can return dict) or Series
    try:
        if "Intercept" in re_series:
            return float(re_series["Intercept"])
    except Exception:
        pass
    try:
        if 0 in re_series:
            return float(re_series[0])
    except Exception:
        pass
    # fallback: first value
    try:
        return float(np.asarray(re_series)[0])
    except Exception:
        return 0.0


def _ci_1sample(values, confidence=0.95):
    """Mean CI for a one-sample vector using the t distribution."""
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(np.mean(values))
    if n < 2:
        return mean, np.nan, np.nan
    sem = stats.sem(values, nan_policy="omit")
    crit = stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, n - 1)
    return mean, float(mean - crit * sem), float(mean + crit * sem)


def _hedges_correction(df):
    """Small-sample correction J(df) for Hedges' g."""
    return 1.0 - (3.0 / (4.0 * df - 1.0))


def mixedlm_r2_nakagawa(mdf):
    """
    Nakagawa-style R2 for a random-intercept Gaussian mixed model.

    Returns marginal R2 (fixed effects only) and conditional R2
    (fixed + random effects).
    """
    exog = np.asarray(mdf.model.exog, dtype=float)
    fe = np.asarray(mdf.fe_params, dtype=float)
    fixed_pred = exog @ fe
    var_fixed = float(np.var(fixed_pred, ddof=1))

    cov_re = np.asarray(mdf.cov_re, dtype=float)
    var_random = float(np.trace(cov_re)) if cov_re.size else 0.0
    var_resid = float(mdf.scale)
    denom = var_fixed + var_random + var_resid

    if denom <= 0:
        return {"r2_marginal": np.nan, "r2_conditional": np.nan}

    return {
        "r2_marginal": var_fixed / denom,
        "r2_conditional": (var_fixed + var_random) / denom,
    }


def _cluster_bootstrap_lme(df, method, index_col, recode_first_to_zero=True,
                           n_boot=500, confidence=0.95, seed=12345):
    """
    Participant-cluster bootstrap CIs for slope and mixed-model R2.
    Duplicate sampled participants are renamed so each bootstrap cluster is independent.
    """
    rng = np.random.default_rng(seed)
    participants = np.asarray(sorted(df.loc[df["method"] == method, "participant"].unique()))
    if participants.size < 2:
        return {}

    rows = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=participants.size, replace=True)
        boot_parts = []
        for sample_i, participant in enumerate(sampled):
            part_df = df[(df["method"] == method) & (df["participant"] == participant)].copy()
            part_df["participant"] = f"{participant}__boot{sample_i}"
            boot_parts.append(part_df)
        boot_df = pd.concat(boot_parts, ignore_index=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, _, mdf = lme_learning_random_intercept(
                    boot_df,
                    method,
                    index_col=index_col,
                    recode_first_to_zero=recode_first_to_zero,
                )
            r2 = mixedlm_r2_nakagawa(mdf)
            rows.append({
                "slope": slope,
                "r2_marginal": r2["r2_marginal"],
                "r2_conditional": r2["r2_conditional"],
            })
        except Exception:
            continue

    if not rows:
        return {}

    boot = pd.DataFrame(rows)
    alpha = (1.0 - confidence) / 2.0
    out = {"n_boot_success": int(len(boot))}
    for col in ["slope", "r2_marginal", "r2_conditional"]:
        lo, hi = np.nanquantile(boot[col], [alpha, 1.0 - alpha])
        out[f"{col}_boot_ci_low"] = float(lo)
        out[f"{col}_boot_ci_high"] = float(hi)
    return out


def lme_learning_summary(df, method, index_col, recode_first_to_zero=True,
                         confidence=0.95, n_boot=500, seed=12345):
    """
    Fit the random-intercept learning model and return slope, slope CI,
    p-value, t-value, and Nakagawa marginal/conditional R2 with bootstrap CIs.
    """
    slope, pval, mdf = lme_learning_random_intercept(
        df, method, index_col=index_col, recode_first_to_zero=recode_first_to_zero
    )
    key = f"{index_col}0" if recode_first_to_zero else index_col
    ci = mdf.conf_int(alpha=1.0 - confidence).loc[key]
    r2 = mixedlm_r2_nakagawa(mdf)
    boot = _cluster_bootstrap_lme(
        df,
        method,
        index_col=index_col,
        recode_first_to_zero=recode_first_to_zero,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )

    out = {
        "method": method,
        "slope": slope,
        "slope_ci_low": float(ci.iloc[0]),
        "slope_ci_high": float(ci.iloc[1]),
        "t_value": float(mdf.tvalues[key]),
        "p_value": pval,
        **r2,
        **boot,
    }
    return out, mdf


def lme_learning_summary_app(df, method, recode_app1_to_zero=True,
                             confidence=0.95, n_boot=500, seed=12345):
    return lme_learning_summary(
        df, method, index_col="app_index", recode_first_to_zero=recode_app1_to_zero,
        confidence=confidence, n_boot=n_boot, seed=seed
    )


def lme_learning_summary_rem(df, method, recode_rem1_to_zero=True,
                             confidence=0.95, n_boot=500, seed=12345):
    return lme_learning_summary(
        df, method, index_col="rem_index", recode_first_to_zero=recode_rem1_to_zero,
        confidence=confidence, n_boot=n_boot, seed=seed
    )


def paired_summary_from_arrays(tt_vals, dt_vals, confidence=0.95):
    """
    Paired TT vs DT summary: mean difference CI, paired Cohen's dz,
    and Hedges' g using the small-sample correction for dz.
    """
    tt_vals = np.asarray(tt_vals, dtype=float)
    dt_vals = np.asarray(dt_vals, dtype=float)
    diff = tt_vals - dt_vals
    n = diff.size
    mean_diff, ci_low, ci_high = _ci_1sample(diff, confidence=confidence)
    tval, pval = stats.ttest_rel(tt_vals, dt_vals)

    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    cohen_dz = mean_diff / sd_diff if sd_diff > 0 else np.nan
    hedges_g = _hedges_correction(n - 1) * cohen_dz if n > 1 else np.nan

    return {
        "n_pairs": int(n),
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_value": float(tval),
        "p_value": float(pval),
        "cohen_dz": float(cohen_dz),
        "hedges_g": float(hedges_g),
    }


def paired_test_collapsed_summary(df, which="app", confidence=0.95):
    """Collapsed paired-test summary comparing TT vs DT across participant x index pairs."""
    idx_col = "app_index" if which == "app" else "rem_index"
    participants = sorted(df["participant"].unique())
    idx_vals = [1, 2, 3]

    dt_vals, tt_vals = [], []
    for p in participants:
        for i in idx_vals:
            dt = df[(df.participant == p) & (df.method == "DT") & (df[idx_col] == i)]["time_sec"].iloc[0]
            tt = df[(df.participant == p) & (df.method == "TT") & (df[idx_col] == i)]["time_sec"].iloc[0]
            dt_vals.append(dt)
            tt_vals.append(tt)

    return paired_summary_from_arrays(tt_vals, dt_vals, confidence=confidence)



# Mixed effects (random intercept only)

def lme_learning_random_intercept(df, method, index_col, recode_first_to_zero=True):
    """
    Fit an LME with random intercepts for participants:
        time_sec ~ (index_col or index_col0) with groups=participant

    Parameters
    ----------
    df : DataFrame with columns: participant, method, <index_col>, time_sec
    method : str, e.g., "TT" or "DT"
    index_col : str, either "app_index" or "rem_index" (or any similar column)
    recode_first_to_zero : bool
        If True, creates "<index_col>0" = <index_col> - 1 and uses that in the model.

    Returns
    -------
    slope : float
    pval  : float
    mdf   : statsmodels MixedLMResults
    """
    d = df[df["method"] == method].copy()

    if recode_first_to_zero:
        key = f"{index_col}0"
        d[key] = d[index_col] - 1
        formula = f"time_sec ~ {key}"
    else:
        key = index_col
        formula = f"time_sec ~ {key}"

    md = smf.mixedlm(formula, d, groups=d["participant"])
    mdf = md.fit(reml=True, method="lbfgs", disp=False)

    slope = float(mdf.params[key])
    pval = float(mdf.pvalues[key])
    return slope, pval, mdf

# Wrappers for APP vs REM 
def lme_learning_random_intercept_app(df, method, recode_app1_to_zero=True):
    return lme_learning_random_intercept(
        df, method, index_col="app_index", recode_first_to_zero=recode_app1_to_zero
    )

def lme_learning_random_intercept_rem(df, method, recode_rem1_to_zero=True):
    return lme_learning_random_intercept(
        df, method, index_col="rem_index", recode_first_to_zero=recode_rem1_to_zero
    )


# Intercepts table printers (APP1 / REM1 baselines per participant)

def lme_intercepts_table(df, method, index_col,
                         recode_first_to_zero=True, mdf=None):
    """
    Return a per-participant baseline intercept table for the given LME.
    If recode_first_to_zero=True, this represents the baseline at index==1 (APP1/REM1).
    Returns:
        DataFrame with columns: participant, method, intercept_sec, intercept_str
    """
    if mdf is None:
        _, _, mdf = lme_learning_random_intercept(
            df, method, index_col=index_col, recode_first_to_zero=recode_first_to_zero
        )

    fixed_intercept = float(mdf.params["Intercept"])

    d = df[df["method"] == method]
    participants = sorted(d["participant"].unique())

    rows = []
    for pid in participants:
        re = mdf.random_effects.get(pid, None)
        offset = _random_intercept_value(re)
        intercept_i = fixed_intercept + offset
        rows.append({
            "participant": pid,
            "method": method,
            "fixed_intercept_sec": fixed_intercept,
            "random_offset_sec": offset,
            "intercept_sec": intercept_i,
            "intercept_str": _sec_to_mmss_simple(intercept_i),
        })

    out = pd.DataFrame(rows).sort_values("participant")
    return out


# Wrappers for APP vs REM 
def lme_intercepts_table_app(df, method, recode_app1_to_zero=True, mdf=None):
    return lme_intercepts_table(
        df, method, index_col="app_index",
        recode_first_to_zero=recode_app1_to_zero, mdf=mdf
    )

def lme_intercepts_table_rem(df, method, recode_rem1_to_zero=True, mdf=None):
    return lme_intercepts_table(
        df, method, index_col="rem_index",
        recode_first_to_zero=recode_rem1_to_zero, mdf=mdf
    )


# Collapsed paired tests: TT vs DT (time data)

def paired_test_collapsed(df, which="app"):
    """Collapsed paired t-test comparing TT vs DT across all participantxindex pairs.
    which = 'app' or 'rem'
    expected columns (APP):
        participant, method, app_index, time_sec
    expected columns (REM):
        participant, method, rem_index, time_sec
    returns:
        mean_diff_sec (TT-DT), t_value, p_value
    """
    idx_col = "app_index" if which == "app" else "rem_index"
    participants = sorted(df["participant"].unique())
    idx_vals = [1,2,3] # fixed at 3 APPs/REMs for now

    dt_vals, tt_vals = [], []
    for p in participants:
        for i in idx_vals:
            dt = df[(df.participant==p)&(df.method=="DT")&(df[idx_col]==i)]["time_sec"].iloc[0]
            tt = df[(df.participant==p)&(df.method=="TT")&(df[idx_col]==i)]["time_sec"].iloc[0]
            dt_vals.append(dt); tt_vals.append(tt)

    tval, pval = stats.ttest_rel(tt_vals, dt_vals)
    mean_diff = float(np.mean(np.array(tt_vals) - np.array(dt_vals)))
    return mean_diff, tval, pval


# Ratings: paired tests per metric (TT vs DT)

def ratings_paired_tests_per_metric(df_rate):
    """For each metric, run a paired t-test TT vs DT across all participantxAPP pairs.
    expected columns:
        participant, method, app_index, metric, rating
    returns:
        DataFrame with columns including metric, mean_diff (TT-DT), CI,
        t_value, p_value, paired Cohen's dz, and Hedges' g.
    """
    out = []
    metrics = ["ease","movement","stability","adhesion","wires","overall"]
    parts = sorted(df_rate["participant"].unique())
    apps = [1,2,3]
    for met in metrics:
        dt_vals, tt_vals = [], []
        for p in parts:
            for a in apps:
                dt = df_rate[(df_rate.participant==p)&(df_rate.method=="DT")&
                             (df_rate.app_index==a)&(df_rate.metric==met)]["rating"].iloc[0]
                tt = df_rate[(df_rate.participant==p)&(df_rate.method=="TT")&
                             (df_rate.app_index==a)&(df_rate.metric==met)]["rating"].iloc[0]
                dt_vals.append(dt); tt_vals.append(tt)
        summary = paired_summary_from_arrays(tt_vals, dt_vals)
        out.append({"metric": met, **summary})
    return pd.DataFrame(out)
