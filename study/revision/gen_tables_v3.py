"""
Generates all tab_*_v3.tex tables for paper_v3.tex from the redesigned methodology's output:
  - study/data/params_maxdepth_only.csv        (reference max_depth, best_score, rho, rho_k)
  - study/data/holdout_results_maxdepth_only.csv (repeated 90/10 holdout, all comparison pairs)

Both classification and regression dataset tables are (re)generated here -- under the old mixed
max_depth/min_samples_leaf grid, classification's table was unaffected by the dataset-pool swap
and was just copied; under max_depth-only, every dataset (classification included) gets a fresh
selection, so both need building from scratch now.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, combine_pvalues

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
OUT_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer'

EXCLUDED_REGR = ['o-ring', 'diabetes']
NEW_REGR = ['concrete', 'boom_bikes']

DATASET_MAP = {
    'lymphography-normal-fibrosis': 'lymphography',
    'stock_portfolio_performance': 'stock-portfolio',
}
CITATIONS = {'concrete': 'uci', 'boom_bikes': 'boombikes'}


def texname(name):
    return DATASET_MAP.get(name, name).replace('_', '-')


sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')
from datasets import binclas_datasets, regr_datasets
import common_datasets.regression as regr_mod

binclas_names = binclas_datasets['name'].tolist()

regr_continuing = regr_datasets[~regr_datasets['name'].isin(EXCLUDED_REGR)]
all_loaders = regr_mod.get_filtered_data_loaders(n_bounds=(1, 1_000_000), n_from_phenotypes=1)
summary = regr_mod.get_summary_pdf()
summary = summary[summary['data_loader_function'].isin(all_loaders)]
new_rows = summary[summary['name'].isin(NEW_REGR)]
regr_names = regr_continuing['name'].tolist() + new_rows.sort_values('n')['name'].tolist()

params = pd.read_csv(f'{DATA_DIR}/params_maxdepth_only.csv')
holdout = pd.read_csv(f'{DATA_DIR}/holdout_results_maxdepth_only.csv')


def ref_row(estimator, name):
    r = params[(params.estimator == estimator) & (params.name == name)]
    return r.iloc[0] if len(r) else None


def holdout_row(estimator, name, pair):
    r = holdout[(holdout.estimator == estimator) & (holdout.name == name) & (holdout.pair == pair)]
    return r.iloc[0] if len(r) else None


def fmt_depth(d):
    if pd.isna(d):
        return r'$\beta$: $\infty$'
    return f'$\\beta$: {int(d)}'


# ---------------------------------------------------------------------------
# Table 1: dataset summary (classification + regression, both from params_maxdepth_only)
# ---------------------------------------------------------------------------
def build_dataset_table(names, dt_label, rf_label, score_label, section_title, tag, extra_col=None):
    ds_table = binclas_datasets if dt_label == 'dtc' else pd.concat(
        [regr_continuing, new_rows], ignore_index=True)
    ds_lookup = ds_table.set_index('name')

    lines = []
    for name in names:
        n = int(ds_lookup.loc[name, 'n'])
        n_col = int(ds_lookup.loc[name, 'n_col'])
        if dt_label == 'dtc':
            n_grid = int(ds_lookup.loc[name, 'n_grid'])
            n_m = int(ds_lookup.loc[name, 'n_minority'])
            cite = ds_lookup.loc[name, 'citation_key']
        else:
            if name in NEW_REGR:
                n_grid = int(np.sum(np.array(new_rows.set_index('name').loc[name, 'grid'])))
                cite = CITATIONS[name]
            else:
                n_grid = int(ds_lookup.loc[name, 'n_grid'])
                cite = ds_lookup.loc[name, 'citation_key']

        extra = f" & {n_m}" if extra_col else ""
        lines.append(f"{texname(name)} & {n} & {n_col} & {n_grid}{extra} \\\\")

    ncols = 4 + (1 if extra_col else 0)
    colspec = "l@{\\hspace{16pt}}" + "r@{\\hspace{16pt}}" * (2 + (1 if extra_col else 0)) + "r"
    nm_col = " & $N_m$" if extra_col else ""
    header = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
              f"\\multicolumn{{{ncols}}}{{c}}{{{section_title}}} \\\\\n"
              f"dataset & N & $N_a$ & $N_b${nm_col} \\\\\n\\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_datasets_{tag}_v3.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_dataset_table(binclas_names, 'dtc', 'rfc', 'auc', 'Classification', 'classification', extra_col=True)
build_dataset_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'Regression', 'regression', extra_col=False)
print("dataset tables written")


# ---------------------------------------------------------------------------
# Table 2: presence (rho, rho_k, absolute scores, corrected p) -- <=_vs_< pair
# ---------------------------------------------------------------------------
def fmt_diff(v):
    if v is None or pd.isna(v) or abs(v) < 1e-9:
        return '0'
    s = f'{v:.0e}'
    return f' {s}' if not s.startswith('-') else s


def fmt_rate(v):
    if v is None or pd.isna(v) or v == 0:
        return '0'
    return f'{v:.2f}'[1:]


def fmt_sci(v):
    if v is None or pd.isna(v):
        return '--'
    s = f'{v:.1e}'
    if s in ('0.0e+00', '-0.0e+00'):
        return '0'
    if s == '1.0e+00':
        return '1'
    return s


def build_presence_table(names, dt_label, rf_label, score_label, section_title, tag):
    lines = []
    for name in names:
        rt, rf_ = ref_row(dt_label, name), ref_row(rf_label, name)
        ht, hf = holdout_row(dt_label, name, '<=_vs_<'), holdout_row(rf_label, name, '<=_vs_<')

        def cell(ref, h):
            rho, rho_k = fmt_rate(ref['rho']), fmt_rate(ref['rho_k'])
            if h is None:
                return f"{rho} & {rho_k} & & & & "
            leq, lt = f"{h['mean_leq']:.3f}", f"{h['mean_lt']:.3f}"
            mark = '$\\ast$' if h['corrected_p'] < 0.05 else ''
            return f"{rho} & {rho_k} & {leq} & {lt} & {fmt_diff(h['mean_diff'])} & {mark}"

        lines.append(f"{texname(name)} & {cell(rt, ht)} & {cell(rf_, hf)} \\\\")

    header = (r"\begin{tabular}{l@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}"
              r"r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c}" + "\n\\toprule\n"
              f"\\multicolumn{{13}}{{c}}{{{section_title}}} \\\\\n"
              r"dataset & \multicolumn{6}{c}{Decision Tree} & \multicolumn{6}{c}{Random Forest} \\" + "\n"
              f" & $\\rho$ & $\\rho_{{k}}$ & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{d}}$ & p$_{{\\neq}}$ & "
              f"$\\rho$ & $\\rho_{{k}}$ & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{d}}$ & p$_{{\\neq}}$ \\\\" + "\n"
              r"\midrule").replace('$$_{', '_{')
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_presence_{tag}_v3.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_presence_table(binclas_names, 'dtc', 'rfc', 'auc', 'Classification', 'classification')
build_presence_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'Regression', 'regression')
print("presence tables written")


# ---------------------------------------------------------------------------
# Table 3: results (avg vs worst, absolute scores, corrected p) -- avg_vs_worst pair
# ---------------------------------------------------------------------------
def build_results_table(names, dt_label, rf_label, score_label, section_title, tag):
    lines = []
    for name in names:
        h_neq_dt = holdout_row(dt_label, name, '<=_vs_<')
        h_avg_dt = holdout_row(dt_label, name, 'avg_vs_worst')
        h_neq_rf = holdout_row(rf_label, name, '<=_vs_<')
        h_avg_rf = holdout_row(rf_label, name, 'avg_vs_worst')

        def cell(h_avg, h_neq):
            if h_avg is None:
                return " & & & & "
            leq, lt, avg = f"{h_avg['mean_leq']:.3f}", f"{h_avg['mean_lt']:.3f}", f"{h_avg['mean_avg']:.3f}"
            neq_mark = '$\\ast$' if (h_neq is not None and h_neq['corrected_p'] < 0.05) else ''
            avg_mark = '$\\ast$' if h_avg['corrected_p'] < 0.05 else ''
            return f"{leq} & {lt} & {avg} & {neq_mark} & {avg_mark}"

        lines.append(f"{texname(name)} & {cell(h_avg_dt, h_neq_dt)} & {cell(h_avg_rf, h_neq_rf)} \\\\")

    header = (r"\begin{tabular}{l@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}"
              r"r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}c}" + "\n\\toprule\n"
              f"\\multicolumn{{11}}{{c}}{{{section_title}}} \\\\\n"
              r"dataset & \multicolumn{5}{c}{Decision Tree} & \multicolumn{5}{c}{Random Forest} \\" + "\n"
              f" & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{avg}}$ & p$_{{\\neq}}$ & p$_{{avg}}$ & "
              f"{score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{avg}}$ & p$_{{\\neq}}$ & p$_{{avg}}$ \\\\" + "\n"
              r"\midrule").replace('$$_{', '_{')
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_results_{tag}_v3.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_results_table(binclas_names, 'dtc', 'rfc', 'auc', 'Classification', 'classification')
build_results_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'Regression', 'regression')
print("results tables written")


# ---------------------------------------------------------------------------
# Table 2+3 combined: presence (rho, rho_k, <=/< scores, p_neq) and evaluation (avg score, p_avg),
# one row per estimator per dataset (decision tree and random forest stacked, not side by side),
# to keep column count manageable while merging the two tables that previously duplicated the
# auc_leq/auc_lt and p_neq columns.
# ---------------------------------------------------------------------------
def build_combined_table(names, dt_label, rf_label, score_label, section_title, tag):
    def row_for(est, name):
        ref = ref_row(est, name)
        h_neq = holdout_row(est, name, '<=_vs_<')
        h_avg = holdout_row(est, name, 'avg_vs_worst')
        if ref is None:
            return f"{texname(name)} & & & & & & & & "
        rho, rho_k = fmt_rate(ref['rho']), fmt_rate(ref['rho_k'])
        if h_neq is None:
            return f"{texname(name)} & {rho} & {rho_k} & & & & & & "
        leq, lt = f"{h_neq['mean_leq']:.3f}", f"{h_neq['mean_lt']:.3f}"
        neq_mark = '$\\ast$' if h_neq['corrected_p'] < 0.05 else ''
        avg = f"{h_avg['mean_avg']:.3f}" if h_avg is not None else ''
        avg_mark = '$\\ast$' if (h_avg is not None and h_avg['corrected_p'] < 0.05) else ''
        return (f"{texname(name)} & {rho} & {rho_k} & {leq} & {lt} & "
                f"{fmt_diff(h_neq['mean_diff'])} & {neq_mark} & {avg} & {avg_mark}")

    lines = []
    lines.append(r"\multicolumn{9}{l}{\emph{Decision Tree}} \\")
    for name in names:
        lines.append(row_for(dt_label, name) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{l}{\emph{Random Forest}} \\")
    for name in names:
        lines.append(row_for(rf_label, name) + r" \\")

    header = (r"\begin{tabular}{l@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}r@{\hspace{3pt}}c}"
              + "\n\\toprule\n"
              f"\\multicolumn{{9}}{{c}}{{{section_title}}} \\\\\n"
              f"dataset & $\\rho$ & $\\rho_{{k}}$ & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & "
              f"{score_label}$_{{d}}$ & p$_{{\\neq}}$ & {score_label}$_{{avg}}$ & p$_{{avg}}$ \\\\" + "\n"
              r"\midrule").replace('$$_{', '_{')
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_combined_{tag}_v3.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_combined_table(binclas_names, 'dtc', 'rfc', 'auc', 'Classification', 'classification')
build_combined_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'Regression', 'regression')
print("combined presence+results tables written")


# ---------------------------------------------------------------------------
# Table 4/5: cross-dataset meta-analysis (single holdout setting -- 4 rows, one per estimator)
# ---------------------------------------------------------------------------
def sign_and_fisher(group):
    n_pos = int((group['mean_diff'] > 0).sum())
    n_neg = int((group['mean_diff'] < 0).sum())
    n_used = n_pos + n_neg
    sign_p = binomtest(n_pos, n_used, 0.5).pvalue if n_used > 0 else np.nan
    p_clipped = group['corrected_p'].clip(lower=1e-300).dropna().values
    fisher_p = combine_pvalues(p_clipped, method='fisher')[1] if len(p_clipped) else np.nan
    nsig = int((group['corrected_p'] < 0.05).sum())
    return n_pos, n_neg, sign_p, fisher_p, nsig


def build_meta_table(pair, out_name):
    est_labels = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}
    lines = []
    for est in ['dtc', 'rfc', 'dtr', 'rfr']:
        group = holdout[(holdout['estimator'] == est) & (holdout['pair'] == pair)]
        n_pos, n_neg, sign_p, fisher_p, nsig = sign_and_fisher(group)
        fisher = fmt_sci(fisher_p)
        sign_s = fmt_sci(sign_p)
        lines.append(f"{est_labels[est]} & {n_pos}/{n_neg} & {sign_s} & {fisher} & {nsig} \\\\")
    header = (r"\begin{tabular}{l@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r}"
              + "\n\\toprule\n"
              r"estimator & wins/losses & sign-test $p$ & Fisher $p$ & $n_{sig}$ \\" + "\n"
              r"\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/{out_name}.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_meta_table('avg_vs_worst', 'tab_meta_v3')
build_meta_table('select_vs_avg', 'tab_meta_select_v3')
print("meta-analysis tables written")


# ---------------------------------------------------------------------------
# Table 6: hit-rate diagnostic (single holdout setting -- 4 rows)
# ---------------------------------------------------------------------------
def build_hitrate_table():
    est_labels = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}
    lines = []
    for est in ['dtc', 'rfc', 'dtr', 'rfr']:
        group = holdout[(holdout['estimator'] == est) & (holdout['pair'] == 'select_hit_rate')
                         & holdout['mean_diff'].notna()]
        if len(group) == 0:
            continue
        mean_hit = group['mean_diff'].mean()
        n_pos = int((group['mean_diff'] > 0.5).sum())
        n_neg = int((group['mean_diff'] < 0.5).sum())
        n_used = n_pos + n_neg
        sign_p = binomtest(n_pos, n_used, 0.5).pvalue if n_used > 0 else np.nan
        lines.append(f"{est_labels[est]} & {len(group)} & {mean_hit:.3f} & "
                     f"{n_pos}/{n_neg} & {fmt_sci(sign_p)} \\\\")
    header = (r"\begin{tabular}{l@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r}"
              + "\n\\toprule\n"
              r"estimator & $n$ datasets & mean hit rate & wins/losses vs.\ 0.5 & sign-test $p$ \\" + "\n"
              r"\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_hitrate_v3.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_hitrate_table()
print("hit-rate table written")
