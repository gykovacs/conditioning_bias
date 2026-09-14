"""
Generates the rebuilt LaTeX tables for paper_v2.tex from validated data:
  - tab_datasets_{classification,regression}_v2.tex  (dataset summary; regression pool updated)
  - tab_presence_{classification,regression}_v2.tex  (rho/rho_k + corrected-p at K=10)
  - tab_results_{classification,regression}_v2.tex   (avg vs worst, corrected-p at K=10)
  - tab_meta_v2.tex                                   (NEW: cross-dataset meta-analysis, all K)

Reuses study/data/{ksweep_results.csv, ksweep_results_k5_reused.csv, meta_analysis_results.csv,
splits_*.csv, params_*.csv} -- no new experiments, just formatting already-validated results.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
OUT_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer'

EXCLUDED_REGR = ['o-ring', 'diabetes']
NEW_REGR = ['concrete', 'boom_bikes']

DATASET_MAP = {
    'lymphography-normal-fibrosis': 'lymphography',
    'stock_portfolio_performance': 'stock-portfolio',
}

CITATIONS = {
    'concrete': 'uci', 'boom_bikes': 'boombikes',
}


def texname(name):
    n = DATASET_MAP.get(name, name)
    return n.replace('_', '-')


# ---------------------------------------------------------------------------
# load K-sweep results (single combined file: all K, absolute scores, select-mode pairs)
# ---------------------------------------------------------------------------
# v3: corrected select_hit_rate (restricted to non-tied folds), confirmed identical to v2 on
# every other pair/column (max abs diff 0.0 on select_vs_avg) -- fully deterministic rerun.
ksweep = pd.read_csv(f'{DATA_DIR}/ksweep_results_v3.csv')


def get_row(estimator, name, pair, k=10):
    sub = ksweep[(ksweep.estimator == estimator) & (ksweep.name == name) & (ksweep.pair == pair)]
    if len(sub) == 0:
        return None
    if k in sub['k'].values:
        return sub[sub['k'] == k].iloc[0]
    return sub.iloc[(sub['k'] - k).abs().argsort().iloc[0]]


def get_p_meandiff(estimator, name, pair, k=10):
    sub = ksweep[(ksweep.estimator == estimator) & (ksweep.name == name) & (ksweep.pair == pair)]
    if len(sub) == 0:
        return None, None
    # prefer exact k, else nearest available (classification per-dataset caps)
    if k in sub['k'].values:
        row = sub[sub['k'] == k].iloc[0]
    else:
        row = sub.iloc[(sub['k'] - k).abs().argsort().iloc[0]]
    return row['corrected_p'], row['mean_diff']


# ---------------------------------------------------------------------------
# dataset pools, in the paper's existing size-sorted order
# ---------------------------------------------------------------------------
import sys
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

print("classification pool:", binclas_names)
print("regression pool:", regr_names)


# ---------------------------------------------------------------------------
# Table: dataset summary (regression only needs rebuilding; classification unchanged -- reuse)
# ---------------------------------------------------------------------------
def build_dataset_table_regression():
    params_dtr = pd.read_csv(f'{DATA_DIR}/params_dtr.csv')
    params_rfr = pd.read_csv(f'{DATA_DIR}/params_rfr.csv')
    ds_full = pd.concat([regr_continuing[['name', 'n', 'n_col']],
                          new_rows[['name', 'n', 'n_col']]], ignore_index=True)
    ds_lookup = ds_full.set_index('name')

    # n_grid (N_b, susceptible feature count) -- reuse from regr_datasets for continuing,
    # compute fresh for new ones using the same criterion already applied by common_datasets
    grid_lookup = {}
    for _, r in regr_continuing.iterrows():
        grid_lookup[r['name']] = r['n_grid']
    new_grid = summary[summary['name'].isin(NEW_REGR)].set_index('name')
    for name in NEW_REGR:
        row = new_grid.loc[name]
        grid_lookup[name] = int(np.sum(np.array(row['grid'])))

    def fmt_param(row_params, n):
        p = eval(row_params, {'np': np})
        if 'min_samples_leaf' in p:
            val = int(np.floor(p['min_samples_leaf'] * n * 0.8))
            return f'$\\alpha$: {val}'
        if 'max_depth' in p:
            return f'$\\beta$: $\\infty$' if p['max_depth'] is None else f'$\\beta$: {p["max_depth"]}'
        return ''

    lines = []
    for name in regr_names:
        n = int(ds_lookup.loc[name, 'n'])
        n_col = int(ds_lookup.loc[name, 'n_col'])
        n_grid = int(grid_lookup[name])
        cite = CITATIONS.get(name, None)
        if cite is None:
            row = regr_datasets[regr_datasets['name'] == name]
            cite = row.iloc[0]['citation_key'] if len(row) else 'uci'
        prow_t = params_dtr[params_dtr['name'] == name].iloc[0]
        prow_f = params_rfr[params_rfr['name'] == name].iloc[0]
        par_t = fmt_param(prow_t['params'], n)
        par_f = fmt_param(prow_f['params'], n)
        r2_t, r2_f = prow_t['r2'], prow_f['r2']
        lines.append(f"{texname(name)} \\citep{{{cite}}} & {n} & {n_col} & {n_grid} & "
                     f"{par_t} & {r2_t:.3f} & {par_f} & {r2_f:.3f} \\\\")

    header = (r"\begin{tabular}{l@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}"
              r"l@{\hspace{6pt}}r@{\hspace{6pt}}l@{\hspace{6pt}}r}" + "\n\\toprule\n"
              r"\multicolumn{8}{c}{Regression} \\" + "\n"
              r"dataset & N & $N_a$ & $N_b$ & par$_t$ & r$^2_t$ & par$_f$ & r$^2_f$ \\" + "\n"
              r"\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_datasets_regression_v2.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_dataset_table_regression()
print("dataset table (regression) written")


# ---------------------------------------------------------------------------
# Table: presence (rho, rho_k, mean_diff, corrected p at K=10) -- classification + regression
# ---------------------------------------------------------------------------
def load_splits(label):
    df = pd.read_csv(f'{DATA_DIR}/splits_{label}.csv').rename(columns={'Unnamed: 0': 'name'})
    df['name'] = df['name'].apply(lambda n: DATASET_MAP.get(n, n))
    df['rate'] = df['n_lattice_splits'] / df['n_splits']
    df['rate_kfold'] = df['n_lattice_splits_kfold'] / df['n_splits_kfold']
    return df.set_index('name')


def fmt_diff(v):
    if v is None or abs(v) < 1e-9:
        return '0'
    s = f'{v:.0e}'
    return f' {s}' if not s.startswith('-') else s


def fmt_rate(v):
    if v is None or v == 0:
        return '0'
    return f'{v:.2f}'[1:]


def build_presence_table(names, dt_label, rf_label, score_label, out_name, section_title, ncols_tag):
    splits_dt = load_splits(dt_label)
    splits_rf = load_splits(rf_label)
    lines = []
    for name in names:
        tname = texname(name)
        # splits csvs keep underscores; match on raw name via DATASET_MAP-normalized key
        key = DATASET_MAP.get(name, name)
        row_dt = splits_dt.loc[key]
        row_rf = splits_rf.loc[key]

        row_dt_k = get_row(dt_label, name, '<=_vs_<', k=10)
        row_rf_k = get_row(rf_label, name, '<=_vs_<', k=10)
        p_dt, m_dt = get_p_meandiff(dt_label, name, '<=_vs_<', k=10)
        p_rf, m_rf = get_p_meandiff(rf_label, name, '<=_vs_<', k=10)

        def abs_scores(r):
            return (f"{r['mean_leq']:.3f}", f"{r['mean_lt']:.3f}") if r is not None else ('', '')

        leq_dt, lt_dt = abs_scores(row_dt_k)
        leq_rf, lt_rf = abs_scores(row_rf_k)

        lines.append(
            f"{tname} & {fmt_rate(row_dt['rate'])} & {fmt_rate(row_dt['rate_kfold'])} & "
            f"{leq_dt} & {lt_dt} & {fmt_diff(m_dt)} & {'$\\ast$' if (p_dt is not None and p_dt < 0.05) else ''} & "
            f"{fmt_rate(row_rf['rate'])} & {fmt_rate(row_rf['rate_kfold'])} & "
            f"{leq_rf} & {lt_rf} & {fmt_diff(m_rf)} & {'$\\ast$' if (p_rf is not None and p_rf < 0.05) else ''} \\\\"
        )

    header = (r"\begin{tabular}{l@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}"
              r"r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c}" + "\n\\toprule\n"
              f"\\multicolumn{{13}}{{c}}{{{section_title}}} \\\\\n"
              r"dataset & \multicolumn{6}{c}{Decision Tree} & \multicolumn{6}{c}{Random Forest} \\" + "\n"
              f" & $\\rho$ & $\\rho_{{k}}$ & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{d}}$ & p$_{{\\neq}}$ & "
              f"$\\rho$ & $\\rho_{{k}}$ & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{d}}$ & p$_{{\\neq}}$ \\\\" + "\n"
              r"\midrule").replace('$$_{', '_{')
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_presence_{ncols_tag}_v2.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_presence_table(binclas_names, 'dtc', 'rfc', 'auc', 'classification', 'Classification', 'classification')
build_presence_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'regression', 'Regression', 'regression')
print("presence tables written")


# ---------------------------------------------------------------------------
# Table: results (avg vs worst, corrected p at K=10) -- classification + regression
# ---------------------------------------------------------------------------
def build_results_table(names, dt_label, rf_label, score_label, ncols_tag, section_title):
    lines = []
    for name in names:
        tname = texname(name)
        row_dt = get_row(dt_label, name, 'avg_vs_worst', k=10)
        row_rf = get_row(rf_label, name, 'avg_vs_worst', k=10)
        p_neq_dt, _ = get_p_meandiff(dt_label, name, '<=_vs_<', k=10)
        p_avg_dt, _ = get_p_meandiff(dt_label, name, 'avg_vs_worst', k=10)
        p_neq_rf, _ = get_p_meandiff(rf_label, name, '<=_vs_<', k=10)
        p_avg_rf, _ = get_p_meandiff(rf_label, name, 'avg_vs_worst', k=10)

        def cell(row, p_neq, p_avg):
            if row is None:
                return " & & & & "
            leq_s, lt_s, avg_s = f"{row['mean_leq']:.3f}", f"{row['mean_lt']:.3f}", f"{row['mean_avg']:.3f}"
            neq_mark = '$\\ast$' if (p_neq is not None and p_neq < 0.05) else ''
            avg_mark = '$\\ast$' if (p_avg is not None and p_avg < 0.05) else ''
            return f"{leq_s} & {lt_s} & {avg_s} & {neq_mark} & {avg_mark}"

        lines.append(f"{tname} & {cell(row_dt, p_neq_dt, p_avg_dt)} & "
                     f"{cell(row_rf, p_neq_rf, p_avg_rf)} \\\\")

    header = (r"\begin{tabular}{l@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}c@{\hspace{3pt}}"
              r"r@{\hspace{3pt}}r@{\hspace{3pt}}r@{\hspace{3pt}}c@{\hspace{3pt}}c}" + "\n\\toprule\n"
              f"\\multicolumn{{11}}{{c}}{{{section_title}}} \\\\\n"
              r"dataset & \multicolumn{5}{c}{Decision Tree} & \multicolumn{5}{c}{Random Forest} \\" + "\n"
              f" & {score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{avg}}$ & p$_{{\\neq}}$ & p$_{{avg}}$ & "
              f"{score_label}$_{{\\leq}}$ & {score_label}$_{{<}}$ & {score_label}$_{{avg}}$ & p$_{{\\neq}}$ & p$_{{avg}}$ \\\\" + "\n"
              r"\midrule").replace('$$_{', '_{')
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_results_{ncols_tag}_v2.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_results_table(binclas_names, 'dtc', 'rfc', 'auc', 'classification', 'Classification')
build_results_table(regr_names, 'dtr', 'rfr', 'r$^2$', 'regression', 'Regression')
print("results tables written")


# ---------------------------------------------------------------------------
# Table: NEW cross-dataset meta-analysis summary -- computed directly from ksweep (v2/v3)
# ---------------------------------------------------------------------------
from scipy.stats import binomtest, combine_pvalues


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
    data = ksweep[(ksweep['pair'] == pair) & (ksweep['k'].isin([5, 10, 15, 20]))]
    est_labels = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}
    lines = []
    for est in ['dtc', 'rfc', 'dtr', 'rfr']:
        for k in [5, 10, 15, 20]:
            group = data[(data['estimator'] == est) & (data['k'] == k)]
            if len(group) == 0:
                continue
            n_pos, n_neg, sign_p, fisher_p, nsig = sign_and_fisher(group)
            fisher = f"{fisher_p:.1e}" if not np.isnan(fisher_p) else '--'
            sign_s = f"{sign_p:.1e}" if not np.isnan(sign_p) else '--'
            lines.append(f"{est_labels[est]} & {k} & {n_pos}/{n_neg} & {sign_s} & {fisher} & {nsig} \\\\")
    header = (r"\begin{tabular}{l@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r}"
              + "\n\\toprule\n"
              r"estimator & $K$ & wins/losses & sign-test $p$ & Fisher $p$ & $n_{sig}$ \\" + "\n"
              r"\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/{out_name}.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_meta_table('avg_vs_worst', 'tab_meta_v2')
build_meta_table('select_vs_avg', 'tab_meta_select_v2')
print("meta-analysis tables written")

# select_hit_rate table (descriptive; NaN rows dropped where n_valid < 2)
def build_hitrate_table():
    data = ksweep[(ksweep['pair'] == 'select_hit_rate') & (ksweep['k'].isin([5, 10, 15, 20]))]
    est_labels = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}
    lines = []
    for est in ['dtc', 'rfc', 'dtr', 'rfr']:
        for k in [5, 10, 15, 20]:
            group = data[(data['estimator'] == est) & (data['k'] == k) & data['mean_diff'].notna()]
            if len(group) == 0:
                continue
            mean_hit = group['mean_diff'].mean()
            n_pos = int((group['mean_diff'] > 0.5).sum())
            n_neg = int((group['mean_diff'] < 0.5).sum())
            n_used = n_pos + n_neg
            sign_p = binomtest(n_pos, n_used, 0.5).pvalue if n_used > 0 else np.nan
            lines.append(f"{est_labels[est]} & {k} & {len(group)} & {mean_hit:.3f} & "
                         f"{n_pos}/{n_neg} & {sign_p:.1e} \\\\")
    header = (r"\begin{tabular}{l@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r@{\hspace{6pt}}r}"
              + "\n\\toprule\n"
              r"estimator & $K$ & $n$ datasets & mean hit rate & wins/losses vs.\ 0.5 & sign-test $p$ \\" + "\n"
              r"\midrule")
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    with open(f'{OUT_DIR}/tab_hitrate_v2.tex', 'w') as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")


build_hitrate_table()
print("hit-rate table written")
