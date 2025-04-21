# Implement methods for assessing the uncertainty and predictive performance of probabilistic grade models
#
# Three categories of measures are implemented:
# A) Variogram ratios
#    - To evaluate local correlation & spatial variability in the predicted mean
# B) Histogram distances
#    - To evaluate global accuracy using probabilistic symmetric Chi2, Jensen-
#      Shannon divergence, Ruzicka distance (similar to Jaccard index based on
#      intersection over union) and Wasserstein earth-moving distance (based on
#      solving a minimum cost transport problem)
# C) Uncertainty-based measures:
#    - Consensus given validation measurements L = |s(mu_hat, sigma_hat | mu_0)|
#    - Quantify the accuracy, precision, goodness and tightness of model
#      uncertainty estimates based on Deutsch-Goovaerts interpretations which
#      consider the true value containment interval, expected and empirical
#      coverage probabilities associated with the conditional distribution
#      function of a probabilistic model.
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#-------------------------------------------------------------------------------


import datetime as dt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import special
from scipy.stats import norm, wasserstein_distance
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from matplotlib import pyplot as plt
from pdb import set_trace as bp

#-------------------------------
# Methods relating to variogram
#-------------------------------
def compute_variogram_stats(vgrams, references, serial="", percentiles=[25,50,75]):
    """
    Compute the variogram ratio for each method with respect to a reference curve
    and report variogram attenuation statistics, typically taken at qL, median and qU.

    Definition: Given a model "m" and reference "ref", the variogram ratio at lag "x"
                is given by r_{m}(x) = variogram_{m}(x) / variogram{ref}(x).
    General interest: Obtaining quantiles (including the median) of r_{m}(x) over x.

    Parameters
    ----------
        vgrams : dict(skgstat.Variogram)
            a stack of semi-variogram objects keyed by a method str
        references : list(str)
            methods that serve as the basis of comparison such as ["GroundTruth(model)"]
        serial : str
            a short description, code or abbreviation for the current experiment
        percentiles : list(float)
            percentiles for variogram ratios, default: lower, middle and upper quartiles.
    """
    methods = list(vgrams.keys())
    assert(all([r in methods for r in references]))
    x, y = [], []
    # Establish valid interval for the lag (x)
    for m in methods:
        x.append(vgrams[m].bins)
        y.append(vgrams[m].experimental)
    lags = np.mean(x, axis=0)
    is_valid = np.isfinite(np.sum(y, axis=0))
    x = lags[is_valid]
    y = np.array(y)[:, is_valid]
    # Compute gamma{method}(x) / gamma{reference}(x) ratios
    # z represents list(dict), it is used to construct a pandas.DataFrame
    z = []
    ratios, stats = {}, {}
    for r in references:
        ratios[r] = {}
        stats[r] = {}
        i_ref = methods.index(r)
        for k, method in enumerate(methods):
            ratios[r][method] = y[k] / y[i_ref]
            stats[r][method] = np.percentile(ratios[r][method], percentiles)

    for k, method in enumerate(methods):
        d = {'method': method, 'serial': serial}
        d['bins'] = '[' + ','.join(['%.6f' % i for i in x]) + ']' if k==0 else ''
        d[f'variogram({r})'] = '[' + ','.join(['%.6f' % i for i in y[k]]) + ']'
        for r in references:
            for j, pct in enumerate(percentiles):
                d[f'p{pct}({r})'] = stats[r][method][j]
            d[f'ratios({r})'] = '[' + ','.join(['%.6f' % i for i in ratios[r][method]]) + ']'
        z.append(d)

    for r in references:
        ratios[r].pop(r, None)
        stats[r].pop(r, None)

    return ratios, stats, pd.DataFrame(z)

#---------------------------------------------------
# Methods relating to histograms / mean predictions
#---------------------------------------------------
def histogram_of(x, y, n_bins=40, legacy_approach=False):
    """
    Apply quantisation to values and obtain bin counts.
    :param x: instances based on model predicted values
    :param y: instances (true values) based on validation measurements
    :return: (representative values, pmf_x, pmf_y)
    """
    bins = np.r_[-np.inf, np.linspace(min(y), max(y), n_bins+1), np.inf]
    # Change: previously, we set argument density=True. But this yields nonsensical
    # results when values fall into the last bin, an open interval [*, inf), the
    # the count or weight becomes nullified. So, we now normalise the histogram
    # counts manually, effectively applying symmetric boundary extension to obtain
    # representative values for (-inf, bins[1]] and [bins[-2], inf) which implies
    # equal length for all quantisation intervals.
    if legacy_approach:
        pmf_x, edges = np.histogram(x, bins, density=True)
        pmf_y, _ = np.histogram(y, bins, density=True)
    else:
        counts_x, edges = np.histogram(x, bins)
        counts_y, _ = np.histogram(y, bins)
        pmf_x = counts_x / sum(counts_x)
        pmf_y = counts_y / sum(counts_y)
    w = np.diff(edges[1:-1])
    centroids = np.mean(np.vstack([edges[1:-2], edges[2:-1]]), axis=0)
    representative_values = np.r_[centroids[0]-w[0], centroids, centroids[-1]+w[-1]]
    return representative_values, pmf_x, pmf_y

def compute_probabilistic_symmetric_chi_square(p, q):
    """
    The probabilistic symmetric chi-square measure (Deza, 2009)
    relates to "triangular discrimination" defined in (Toesoe, 2000)
    D_{psChi} = 2 * sum_x [|p(x)-q(x)|^2 / (p(x)+q(x))]
    :param p: empirical (observed) probability mass function
    :param q: theoretical model or true probability mass function
    """
    p_ = np.array(p) / sum(p)
    q_ = np.array(q) / sum(q)
    numerator = (p_ - q_)**2
    denominator = 0.5 * (p_ + q_)
    included = np.where(denominator > 0)[0]
    d_psChi2 = sum(numerator[included] / denominator[included])
    return d_psChi2

def compute_jensen_shannon_divergence(p, q):
    """
    The Jensen-Shannon divergence measure, D_{JS}, represents the symmetric form
    of K divergence. It is defined by (1/2)*[sum_x p(x)*log(2*p(x)/(p(x)+q(x)) +
    sum_x q(x)*log(2*q(x)/(p(x)+q(x))] or (1/2)*[KL(p|m) + KL(q|m)] with m=(p+q)/2
    in terms of the Kullback-Leibler distance KL(p|q) = sum_x p(x)*log(p(x)/q(x)).
    Using base-2 logarithm, 0 <= D_{JS} <= 1, attaining zero when p = q.
    Comment: The scipy built-in fn handles unity-sum normalisation and histogram
             bins with zero count. Note: It actually returns the sqrt of D_{JS}.
    """
    return jensenshannon(p, q, base=2)**2

def compute_ruzicka_distance(p, q):
    """
    The distance D_{Ruz} = 1 - S_{Ruz} is defined by Ruzicka similarity S_{Ruz},
    where S_{Ruz} = sum_x min{p(x),q(x)} / sum_x max{p(x),q(x)} given two probability
    mass functions, p and q. This may be interpreted as the intersection(p,q) over
    union(p,q) as a generalisation of the Jaccard index from {0,1}^n to R^n.
    Ruzicka distance is bounded by 0 <= D(Ruz) <= 1
    """
    p_ = np.array(p) / sum(p)
    q_ = np.array(q) / sum(q)
    numerator = np.min(np.c_[p_, q_], axis=1)
    denominator = np.max(np.c_[p_, q_], axis=1)
    included = np.where(denominator > 0)[0]
    similarity_ruz = sum(numerator[included]) / sum(denominator[included])
    return 1 - similarity_ruz

def compute_wasserstein_distance(p_weights, q_weights, p_values=None, q_values=None,
                                 x=None, y=None, verbose=False):
    """
    Compute the Wasserstein-1 distance between two 1D discrete distributions.
    The Wasserstein distance, also called the Earth mover’s distance or the optimal
    transport distance, is a similarity metric that may be interpreted as the minimum
    energy cost of moving and transforming a pile of dirt in the shape of one
    probability distribution into the other. The cost is quantified by the distance
    and amount of probability mass being moved. It might be preferred over JS
    divergence, as the Kantorovich-Mallows-Monge-Wasserstein metric represents the
    Lipschitz distance between probability measures and has to be K-Lipschitz continuous.
    When the measures are uniform over a set of discrete elements, the problem is
    also known as minimum weight bipartite matching. For formal definitions, see
    https://en.m.wikipedia.org/wiki/Earth_mover's_distance

    :param p_weights: empirical (observed) probability mass function
    :param q_weights: theoretical model or true probability mass function
    :param p_values: discrete values (histogram bins) that define the p PMF
    :param q_values: discrete values (histogram bins) that define the q PMF
    :param x: instances that give rise to p (with or without value quantisation)
    :param y: instances that give rise to q (with or without value quantisation)
    """
    if p_values is None:
        p_values = np.linspace(0,1,len(p_weights))
    if q_values is None:
        q_values = np.linspace(0,1,len(q_weights))
    d_scipy = wasserstein_distance(p_values, q_values, p_weights, q_weights)

    # demonstrate in 1D case, it can be calculated using order statistics
    if x is not None and y is not None and len(x) == len(y):
        X_i = np.sort(x.flatten())
        Y_i = np.sort(y.flatten())
        # handle invalid/missing values by random eviction
        X_i = X_i[np.isfinite(X_i)]
        Y_i = Y_i[np.isfinite(Y_i)]
        nx_ny = len(X_i) - len(Y_i)
        if nx_ny != 0:
            np.random.seed(len(x) + int(np.mean(X_i) * 100) % 10)
        if nx_ny > 0:
            evict = np.random.choice(len(X_i), size=nx_ny, replace=False)
            X_i = X_i[np.setdiff1d(np.arange(len(X_i)), evict)]
        else:
            evict = np.random.choice(len(Y_i), size=-nx_ny, replace=False)
            Y_i = Y_i[np.setdiff1d(np.arange(len(Y_i)), evict)]
        # compute step
        exponent, n = 1, len(X_i)
        d_order_stats = (1/n)**(1/exponent) * np.linalg.norm(X_i - Y_i, ord=exponent)
        if verbose:
            print(f"W_p computed using order stats: {d_order_stats}")
        if all([x_ in np.unique(p_values) for x_ in np.unique(x)]) and \
           all([y_ in np.unique(q_values) for y_ in np.unique(y)]):
                assert(np.isclose(d_order_stats, d_scipy))
        else: #x and y are not quantised, we prefer this over the quantised version
            return d_order_stats

    return d_scipy

def compute_histogram_statistics(mu_0, mu_hat, intervals=40):
    """
    Compute various histogram statistics for a probabilistic model in one go
    :param mu_0: true values (vector of m validation measurements)
    :param mu_hat: model predicted mean, shape=(m,)
    :return: dict with keys ['psym-chi-square', 'jensen-shannon', 'ruzicka', 'wasserstein']
    """
    values, pmf_x, pmf_y = histogram_of(x=mu_hat, y=mu_0, n_bins=intervals)
    d_psChi2 = compute_probabilistic_symmetric_chi_square(pmf_x, pmf_y)
    d_js = compute_jensen_shannon_divergence(pmf_x, pmf_y)
    d_ruz = compute_ruzicka_distance(pmf_x, pmf_y)
    d_em = compute_wasserstein_distance(pmf_x, pmf_y, values, values, x=mu_hat, y=mu_0)
    return {'psym-chi-square': d_psChi2,
            'jensen-shannon': d_js,
            'ruzicka': d_ruz,
            'wasserstein': d_em,
            'values': values,
            'pmf_x': pmf_x,
            'pmf_y': pmf_y
            }

def compute_root_mean_squared_error(mu_0, mu_hat):
    return np.sqrt(np.nanmean((mu_hat - mu_0)**2))
  
#-----------------------------------------------------------
# Methods relating to uncertainty / p-probability intervals
#-----------------------------------------------------------
def compute_synchronicity(mu_0, mu_hat, sigma_hat, omit_z_scores=True, eps=1e-6):
    """
    Assess how reasonable the cdf model parameters are given the ground truth
    :param mu_0: true mean (vector of m validation measurements)
    :param mu_hat: estimated mean, shape=(m,)
    :param sigma_hat: estimated standard deviation, shape=(m,)
    :return: signed scores "s" in [-1,1] where
             magnitude |s| gives the local consensus,
             negative indicates over-estimation (mu_0 < mu_hat),
             positive indicates under-estimation (mu_hat < mu_0)
    """
    # Compute the z-score. These may be interpreted as quantiles of the standard normal distribution
    z_scores = (mu_0 - mu_hat) / (sigma_hat + eps)
    under_estimated = mu_0 > mu_hat
    # Define s := | [0.5 - (Phi(z) - Phi(0))] * 2, if mu_0 > mu_hat
    #             | - Phi(z) * 2                 , otherwise
    s = 2 * (under_estimated * (1 - norm.cdf(z_scores))
          - (1 - under_estimated) * norm.cdf(z_scores))
    return s if omit_z_scores else (s, z_scores)

def compute_model_consensus(mu_0, mu_hat, sigma_hat, n_sigma=None, alpha=None):
    """
    Compute local consensus L = |a(mu_hat, sigma_hat | mu_0)| using validation data.
    Assess how reasonable the (mu_hat, sigma_hat) estimates are given the true values.

    Parameters
    ----------
        mu_0 : numpy.ndarray, shape=(n,) or float
            true values at the inference locations (e.g. Cu grade value for each block)
        mu_hat : numpy.ndarray, shape=(n,) or float
            mean estimates obtained from a model such as GP-SGS or CRF
        sigma_hat: numpy.ndarray, shape=(n,) or float
            standard deviation estimates obtained from the model
        n_sigma : list(float)
            when a list is given, it returns tail statistics frac[p(z) < t]
        alpha : list(float), typical values are 0.95, 0.99
            option to automatically deduce n_sigma for significance testing
    """
    s_scores, z_scores = compute_synchronicity(
                         mu_0, mu_hat, sigma_hat, omit_z_scores=False)
    consensus = p_c = np.abs(s_scores)
    # Compute tail statistics for various multipliers of sigma (m)
    if alpha is not None:
        n_sigma = norm.ppf(1 - (1 - np.array(alpha))/2, 0, 1)
    if n_sigma is not None:
        # if n_sigma = [0.5, 1, 1.5, 2, 2.5, 3],
        #    t = [0.6170, 0.3173, 0.1336, 0.0455, 0.0124, 0.0027]
        # if alpha = [0.68, 0.9, 0.95, 0.975, 0.99],
        #    n_sigma = [0.994457, 1.644853, 1.959963, 2.241402, 2.575829]
        #    t = [0.32, 0.1, 0.05, 0.025, 0.01]
        tail_stats = {}
        # Report [n_sigma, significance(t), observed proportion]
        for m in n_sigma:
            t = 1 - 2 * (norm.cdf(m, 0, 1) - 0.5)
            tail_stats[m] = [m, t, sum(p_c <= t)/len(p_c)]
        return consensus, s_scores, z_scores, tail_stats
    else:
        return consensus, s_scores, z_scores

def compute_kappa(s_scores, p, slack=0.):
    """
    Compute the fraction of true values falling within the symmetric p-probability intervals (PI)
    given validation measurements (true values) of the variable at locations {x(j)}, j=1:m.
    Given p, the p-probability interval is given by [pL,pU] where pL=(1-p)/2, pU=(1+p)/2.

    Task: Compute kappa_bar(p) = A = (1/m) * sum_{j=1:m} kappa_j(p)
          where kappa_j(p) = 1  if QL(j,p) < mu_0(j) < QU(j,p), 0 otherwise
                QL(j,p) and QU(j,p) denote the pL and pU quantiles of the estimated
                conditional distribution function (cdf) for a random variable of
                interest (Y) which may be written as p(Y < y | neighbourhood(X_j)).

    Using Z-score transformation, with z_0(j) = (mu_0(j) - mu_hat(j)) / sigma_hat(j),
                kappa_j(p) = 1  if qL(j,p) < z_0(j) < qU(j,p), 0 otherwise.
    Further, qL = -qU when the random function is Gaussian distribution.

    Instead of sweeping over p, there's an easy way to test this. Since, there exists a
    critical value p* where z_0(j) lies just on the edge of [qL(j,p*), qU(j,p*)].
    The `compute_model_consensus` function serves this purpose, it converts each [mu_0(j),
    mu_hat(j), sigma_hat(j)] into a z-score that corresponds to qL(j,*) or qU(j,p*),
    which gets mapped to a consensus value |s*| = 1 - p*.
    Since the interval grows with p, kappa_j(p) = 1 for all p >= p*, where p* = 1 - |s*|
    """
    p_star = 1.0 - np.abs(s_scores)
    m = len(s_scores)
    kappa_bar = sum(p >= (1 - slack) * p_star) / m
    return kappa_bar

def compute_kappa_bar(s_scores, p_values, slack=0.):
    """
    Compute accuracy of the estimated distribution for various p
    :param s_scores: signed scores produced by `compute_synchronicity`, shape=(m,)
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
                     for instance, np.r_[np.linspace(0,1,41)[1:-1], 0.9825, 0.99, 0.997]
    :param slack: slack variable (zero or a small positive value << 1)
    """
    kappa_bars = np.zeros(len(p_values))
    for i, p in enumerate(p_values):
        kappa_bars[i] = compute_kappa(s_scores, p, slack)
    return kappa_bars

def compute_kappa_mean_proportion(p_values, kappa_bar=None, s_scores=None, slack=0.):
    """
    Compute mean proportion using kappa_bar based on the notion of p-probability intervals
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    :param kappa_bar: accuracy scores produced by `compute_kappa_bar` (optional)
    :param s_scores: signed scores produced by `compute_synchronicity` (required if `kappa_bar` is missing)
    :param slack: slack variable (zero or a small positive value << 1).
    """
    if kappa_bar is None:
        kappa_bar = compute_kappa_bar(s_scores, p_values, slack)
    return np.mean(kappa_bar)
    
def compute_distribution_accuracy(p_values, kappa_bar=None, s_scores=None, slack=0.):
    """
    Compute average accuracy of the estimated distribution
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    :param kappa_bar: accuracy scores produced by `compute_kappa_bar` (optional)
    :param s_scores: signed scores produced by `compute_synchronicity` (required if `kappa_bar` is missing)
    :param slack: slack variable (zero or a small positive value << 1).
    """
    if kappa_bar is None:
        kappa_bar = compute_kappa_bar(s_scores, p_values)
    indicator = kappa_bar >= (1 - slack) * p_values
    return np.mean(indicator)

def compute_uncertainty_precision(p_values, kappa_bar=None, s_scores=None):
    """
    Measure closeness of kappa_bar(p) to p when distribution is accurate at p.
    If p is uniformly spaced, then P = 1 - 2 * mean(indicator * (kappa_bar - p_values))
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    :param kappa_bar: accuracy scores produced by `compute_kappa_bar` (optional)
    :param s_scores: signed scores produced by `compute_synchronicity` (required if `kappa_bar` is missing)
    """
    if kappa_bar is None:
        kappa_bar = compute_kappa_bar(s_scores, p_values)
    indicator = kappa_bar >= p_values
    dp = p_values - np.r_[0, p_values[:-1]]
    dp /= sum(dp) #normalisation
    return 1 - 2 * sum(indicator * (kappa_bar - p_values) * dp)

def compute_uncertainty_goodness_statistic(p_values, kappa_bar=None, s_scores=None):
    """
    Compute C.V. Deutsch's goodness statistic
        G = 1 - integral_{0}^{1} [3*indicator(p) - 2] [kappa_bar(p) - p] dp
    using either kappa_bar(p) := the estimated proportions of coverage by the model's prediction
    uncertainty where the interval is parameterised by p, OR synchronicity (s_scores).
    Unlike accuracy and precision, this takes into account instances where kappa_bar(p) < p.
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    :param kappa_bar: accuracy scores produced by `compute_kappa_bar` (optional)
    :param s_scores: signed scores produced by `compute_synchronicity` (required if `kappa_bar` is missing)
    """
    if kappa_bar is None:
        kappa_bar = compute_kappa_bar(s_scores, p_values)
    dp = p_values - np.r_[0, p_values[:-1]]
    dp /= sum(dp) #normalisation
    indicator = kappa_bar >= p_values
    integral = sum((3 * indicator - 2) * (kappa_bar - p_values) * dp)
    return 1 - integral

def mean_prediction_uncertainty_width(sigma_hat, s_scores, p):
    """
    Compute average width of prediction uncertainty intervals for probability p (Goovaerts, 2001)
    :param sigma_hat: estimated standard deviation, shape=(m,)
    :param s_scores: signed scores produced by `compute_synchronicity`, shape=(m,)
    :param p: a probability value
    """
    #warnings.filterwarnings("error")
    m = len(s_scores)
    p_star = 1.0 - np.abs(s_scores)
    pL = (1 - p_star)/2.
    pU = (1 + p_star)/2.
    qL = np.maximum(norm.ppf(pL, loc=0, scale=1), -5.0)
    qU = np.minimum(norm.ppf(pU, loc=0, scale=1), +5.0)
    kappa_j = p >= p_star
    kappa_bar = compute_kappa(s_scores, p)
    # scaling depends on sigma_hat but is independent of mu_hat
    # As Q(p) = q(p) * sigma_hat + mu_hat, |QU - QL| = |qU - qL| * sigma_hat
    w_bar = sum(kappa_j * (qU - qL) * sigma_hat) / (m * kappa_bar) if kappa_bar > 0 else np.nan

    return w_bar

def compute_width_bar(sigma_hat, s_scores, p_values):
    """
    Compute average width of prediction uncertainty intervals for various p.
    :param sigma_hat: estimated standard deviation, shape=(m,)
    :param s_scores: signed scores produced by `compute_synchronicity`, shape=(m,)
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    """
    w_bar_p = np.zeros(len(p_values))
    for i, p in enumerate(p_values):
        w_bar_p[i] = mean_prediction_uncertainty_width(sigma_hat, s_scores, p)
    return w_bar_p

def compute_uncertainty_tightness_statistic(p_values, s_scores, mu_0, sigma_hat):
    """
    Compute tightness of uncertainty interval over p
    :param p_values: a dense array of probability values from 0 to 1 (excluding 0 and 1)
    :param s_scores: signed scores produced by `compute_synchronicity`, shape=(m,)
    :param mu_0: true mean (vector of m validation measurements)
    :param sigma_hat: estimated standard deviation, shape=(m,)
    """
    sigma_y = np.std(mu_0)
    w_bar_p = compute_width_bar(sigma_hat, s_scores, p_values)
    scaling = sigma_y if sigma_y > 0 else 1.0
    w_bar_normalised = w_bar_p / scaling
    idx = np.isfinite(w_bar_p)
    dp = p_values - np.r_[0, p_values[:-1]]
    integral = sum(w_bar_normalised[idx] * dp[idx]) / sum(dp[idx]) if any(idx) else np.nan
    return integral

def compute_performance_statistics(mu_0_in, mu_hat_in, sigma_hat_in, p_values=None, slack=0.):
    """
    Compute various performance statistics for a probabilistic model in one go
    :return: dict with keys ['s_scores', 'consensus', 'accuracy', 'precision', 'goodness', 'tightness']
    """
    # handle invalid/missing values
    retain = np.isfinite(mu_0_in + mu_hat_in + sigma_hat_in)
    mu_0 = mu_0_in[retain]
    mu_hat = mu_hat_in[retain]
    sigma_hat = sigma_hat_in[retain]
    
    if p_values is None:
        #previous: p = np.r_[np.linspace(0,1,41)[1:-1], 0.9825, 0.99, 0.997]
        segment1 = np.linspace(0,0.98,247)[1:-1]
        segment2 = np.linspace(0.98,1,12)[:-1]
        p_values = np.r_[segment1, segment2]

    consensus, s_scores = compute_model_consensus(mu_0, mu_hat, sigma_hat)[:2]
    kappa_bar = compute_kappa_bar(s_scores, p_values)
    proportion = compute_kappa_mean_proportion(p_values, None, s_scores, slack)
    accuracy = compute_distribution_accuracy(p_values, None, s_scores, slack)
    precision = compute_uncertainty_precision(p_values, kappa_bar)
    goodness = compute_uncertainty_goodness_statistic(p_values, kappa_bar)
    width = compute_width_bar(sigma_hat, s_scores, p_values)
    tightness = compute_uncertainty_tightness_statistic(p_values, s_scores, mu_0, sigma_hat)
    return {'s_scores': s_scores,
            'consensus': consensus,
            'proportion': proportion,
            'accuracy': accuracy,
            'precision': precision,
            'goodness': goodness,
            'width': width,
            'tightness': tightness,
            'invalid_samples': np.where(retain==False)[0]
            }
