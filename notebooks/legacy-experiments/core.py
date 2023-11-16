
import tqdm

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

__all__ = ['evaluate',
            'evaluate_scenarios_regression',
            'evaluate_scenarios_classification',
            'do_comparisons']

def evaluate(scenarios,
                compare,
                data_loaders,
                validator_params,
                score,
                random_state=None):
    results = []
    
    names_means = [f"{score}_mean_{scenario['name']}" for scenario in scenarios]
    names_scores = [f"{score}s_{scenario['name']}" for scenario in scenarios]
    names_comparisons = [f"p_{comparison[0]}_{comparison[1]}" for comparison in compare]

    for data_loader in data_loaders:
        dataset = data_loader()
        X = dataset['data']
        y = dataset['target']

        if score == 'r2':
            raw_scores = evaluate_scenarios_regression(scenarios, X, y, validator_params, random_state=random_state)
        elif score == 'auc':
            raw_scores = evaluate_scenarios_classification(scenarios, X, y, validator_params, random_state=random_state)

        mean_scores = [np.mean(raw_scores[scenario['name']]) for scenario in scenarios]
        scores = [raw_scores[scenario['name']] for scenario in scenarios]

        comparisons = do_comparisons(raw_scores, compare)

        results.append([dataset['name']] + mean_scores + scores + comparisons)

        tmp = pd.DataFrame([results[-1]], columns=['name'] + names_means + names_scores + names_comparisons)
        print(tmp[['name'] + names_means + names_comparisons])

    return pd.DataFrame(results, columns=['name'] + names_means + names_scores + names_comparisons)

def evaluate_scenarios_regression(scenarios, X, y, validator_params, random_state=None):
    validator = RepeatedKFold(**validator_params)

    r2s = {scenario['name']: [] for scenario in scenarios}
    
    if random_state is not None:
        random_state = np.random.RandomState(random_state)

    for train, test in tqdm.tqdm(validator.split(X, y)):
        random_seed = None
        if random_state is not None:
            random_seed = random_state.randint(validator_params['n_repeats'] * validator_params['n_splits'] * 100)
        
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        for scenario in scenarios:
            estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
            estimator.fit(X_train * scenario['multiplier'], y_train)
            pred = estimator.predict(X_test * scenario['multiplier'])
            r2s[scenario['name']].append(r2_score(y_test, pred))

    return r2s

def evaluate_scenarios_classification(scenarios, X, y, validator_params, random_state=None):
    validator = RepeatedStratifiedKFold(**validator_params)

    aucs = {scenario['name']: [] for scenario in scenarios}

    if random_state is not None:
        random_state = np.random.RandomState(random_state)

    for train, test in tqdm.tqdm(validator.split(X, y)):
        random_seed = None
        if random_state is not None:
            random_seed = random_state.randint(validator_params['n_repeats'] * validator_params['n_splits'] * 100)
        
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        for scenario in scenarios:
            estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
            estimator.fit(X_train * scenario['multiplier'], y_train)
            pred = estimator.predict_proba(X_test * scenario['multiplier'])[:, 1]
            aucs[scenario['name']].append(roc_auc_score(y_test, pred))

    return aucs

def do_comparisons(scores, compare):
    comparisons = []
    for comparison in compare:
        comparisons.append(wilcoxon(scores[comparison[0]],
                                    scores[comparison[1]],
                                    alternative='less',
                                    zero_method='zsplit').pvalue)

    return comparisons
