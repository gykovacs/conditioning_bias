"""
Synthetic data generator for the conditioning-bias study (validation plan section 2a).

Design goals (from reviews 1 & 2, and to give a controlled testbed for reviewer 4's claims):
  - a controllable discrete/equidistant feature (the one susceptible to conditioning bias),
    alongside continuous "filler" features that are NOT susceptible;
  - a known data-generating process, so we can draw a near-infinite i.i.d. test set and measure
    the TRUE population-level effect of the conditioning operator directly -- no CV, no repeated
    folds, no non-independence correction needed, because fresh i.i.d. draws are genuinely
    independent;
  - controllable knobs: domain granularity of the discrete feature (how many distinct equidistant
    values), training sample size, and (via the caller's tree hyperparameters) regularization --
    to run a sensitivity study.

The target function is deliberately built so that the discrete feature's exact value matters (not
just its rank), which is what creates the opportunity for thresholds to land exactly on domain
values during tree induction.
"""
import numpy as np


def make_conditioning_bias_regression(n_samples, *, n_continuous=3, discrete_domain_size=11,
                                       discrete_step=1.0, discrete_weight=1.0,
                                       continuous_weight=1.0, noise_std=0.5, random_state=None):
    """
    Generates (X, y) for regression, where the last column of X is the discrete/equidistant
    feature (domain {0, step, 2*step, ..., (domain_size-1)*step}) and the rest are continuous.

    y = continuous_weight * sum(continuous features) + discrete_weight * discrete_feature + noise

    Returns X (n_samples, n_continuous + 1), y (n_samples,).
    """
    rng = np.random.RandomState(random_state)
    X_cont = rng.normal(size=(n_samples, n_continuous))
    discrete_levels = np.arange(discrete_domain_size) * discrete_step
    X_disc = rng.choice(discrete_levels, size=n_samples)

    y = continuous_weight * X_cont.sum(axis=1) + discrete_weight * X_disc
    y += rng.normal(scale=noise_std, size=n_samples)

    X = np.column_stack([X_cont, X_disc])
    return X, y


def make_conditioning_bias_classification(n_samples, *, n_continuous=3, discrete_domain_size=11,
                                           discrete_step=1.0, discrete_weight=1.0,
                                           continuous_weight=1.0, noise_std=0.5,
                                           random_state=None):
    """
    Same generative structure as the regression version, but the continuous latent score is
    thresholded (at its own median, computed from a large reference sample) to produce a
    roughly-balanced binary label.
    """
    rng = np.random.RandomState(random_state)
    X_cont = rng.normal(size=(n_samples, n_continuous))
    discrete_levels = np.arange(discrete_domain_size) * discrete_step
    X_disc = rng.choice(discrete_levels, size=n_samples)

    score = continuous_weight * X_cont.sum(axis=1) + discrete_weight * X_disc
    score += rng.normal(scale=noise_std, size=n_samples)

    # threshold at the theoretical median of the noiseless score's distribution (0, by symmetry
    # of X_cont and X_disc after centering) -- center X_disc so the threshold is meaningful
    X_disc_centered = X_disc - discrete_levels.mean()
    score_centered = continuous_weight * X_cont.sum(axis=1) + discrete_weight * X_disc_centered
    score_centered += rng.normal(scale=noise_std, size=n_samples)
    y = (score_centered > 0).astype(int)

    X = np.column_stack([X_cont, X_disc])
    return X, y


if __name__ == '__main__':
    # sanity check: does the discrete feature actually create lattice-aligned thresholds?
    import sys
    sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_conditioning_bias_regression(500, discrete_domain_size=11, random_state=0)
    tree = DecisionTreeRegressor(max_depth=8, random_state=0).fit(X, y)

    disc_col = X.shape[1] - 1
    disc_levels = set(np.unique(X[:, disc_col]))
    thresholds_on_disc_feature = tree.tree_.threshold[tree.tree_.feature == disc_col]
    on_lattice = [t for t in thresholds_on_disc_feature if any(abs(t - lv) < 1e-9 for lv in disc_levels)]
    print(f"n nodes splitting on discrete feature: {len(thresholds_on_disc_feature)}")
    print(f"n of those with threshold exactly on a domain value: {len(on_lattice)}")
    print(f"discrete feature value counts: {np.unique(X[:, disc_col], return_counts=True)}")
