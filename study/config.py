import os

# dataset specifications
n_datasets = 20
n_bounds_binclas = (1, 1000)
n_minority_bounds_binclas = (5, 1000)
n_bounds_regr = (1, 2000)
n_from_phenotypes = 1

# cross-validation specification
n_splits = 5
n_repeats = 400

# cross-validation for model selection
n_splits_ms = 5
n_splits_ms = 20

# the random seed to use
random_seed = 5

# shortening long dataset names
dataset_map = {
    'lymphography-normal-fibrosis': 'lymphography',
    'stock_portfolio_performance': 'stock-portfolio'
}

# the output directories
data_dir = 'data'
figure_dir = 'figures'
tab_dir = 'tabs'

# creating the output directories if they does not exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

if not os.path.exists(tab_dir):
    os.mkdir(tab_dir)
