#!/usr/bin/env python
# coding: utf-8

# Geostatistical Analysis of Uncertainty and Prediction Performance in Probabilistic Models
# 
# This script provides a template for running the experiments for a given domain and inference period.
# - The `process` API wraps around the `construct_models` and `analyse_models` methods.
# - `construct_models` generates all models of interest:
#    SK, OK, SK-SGS, OK-SGS, GP(L), GP-SGS, GP(R), GP-CRF.
# - `analyse_models` computes the histogram, variogram and uncertainty-based statistics
#    and performs rank analysis. It saves results to csv or pickle files and produces
#    graphics that include blastholes grade visualisation, histograms and variograms,
#    kappa-accuracy and interval tightness plots, model predicted mean and variance,
#    as well as maps that depict uncertainty likelihood and signed distortion.
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------


import matplotlib
matplotlib.use('Agg') #for non-interactive backend, only writing to file

import ast
import copy
import numpy as np
import pandas as pd
import skgstat as skg
import os
import pickle
import sys
import time
import warnings

from collections import OrderedDict
from functools import partial
from matplotlib import pyplot as plt
from pdb import set_trace as bp
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

from gstatsim3d_utils import make_scatter_2d, timeit
from gstatsim3d_gaussian_process import GPManager
from gstatsim3d_kriging import KrigingRegression, KrigingManager

from rtc_trend_alignment import compute_any_rotation_and_scaling_matrix
from rtc_evaluation_metrics import *
from rtc_utils import *


create_scatter_plot = partial(make_scatter_2d, interactive=False)

#---------------------
# Auxiliary functions
#---------------------
def configure_paths_and_conventions(inference_prefix, domain_id, specs):
    mA = inference_prefix
    mA_mB_mC = '%02d_%02d_%02d' % (mA, mA+1, mA+2)
    cfg_rs = dict()
    default_code_dir = os.getcwd()
    default_data_dir = default_code_dir.replace('code', 'data')
    default_result_dir = default_code_dir.replace('code', 'results')
    cfg_rs['info:data_dir'] = specs.get('info:data_dir', default_data_dir)
    cfg_rs['info:result_dir'] = specs.get('info:result_dir', default_result_dir)
    cfg_rs['gp:associate_plunge_rotation_with_x'] = False
    data_path = f"{cfg_rs['info:data_dir']}/{mA_mB_mC}"
    # Check if variogram fitting and GP learning occur in rotated space
    subdir = check_learning_rotation_status(specs)
    result_path = f"{cfg_rs['info:result_dir']}/{subdir}/{mA_mB_mC}"
    figure_path = f"{cfg_rs['info:result_dir']}/{subdir}/{mA_mB_mC}/figs"
    os.makedirs(figure_path, exist_ok=True)
    specs.update(cfg_rs)
    specs.update({'mA': mA, 'domain_id': domain_id})
    return specs, data_path, subdir, result_path, figure_path

def feedback_on_config_overrides(substitutions, inference_prefix, domain_id):
    print(f"Config overrides for mA={inference_prefix}, domain={domain_id}")
    for k, v in substitutions.items():
        type_level_1 = type(substitutions[k])
        type_level_2 = None
        type_level_3 = None
        expand = True
        if isinstance(v, list):
            try:
                type_level_2 = '{}'.format(type(v[0]))
            except:
                pass
        else:
            expand = False
        if expand and isinstance(v[0], list):
            try:
                type_level_3 = '{}'.format(type(v[0][0]))
            except:
                pass
        inner2 = '' if type_level_3 is None else '({})'.format(type_level_3)
        inner1 = '' if type_level_2 is None else '({}{})'.format(type_level_2, inner2)
        outer = '{}{}'.format(type_level_1, inner1)
        type_info = outer.replace("<class '", "").replace("'>", "")
        print(f'- "{k}": type={type_info}, value={v}')

def reorder_optimised_parameters(inference_prefix, domain_id, specs={}):
    specs = configure_paths_and_conventions(inference_prefix, domain_id, specs)[0]
    cfg = {'kriging': create_kriging_config(None, None, specs),
           'gp': create_gaussian_process_config(specs)}
    default_csv = {'kriging': 'gstatsim3d_optimised_parameters_kriging.csv',
                   'gp': 'gstatsim3d_optimised_hyperparameters_gp.csv'}
    for cat in ['kriging', 'gp']:
        file = os.path.join(cfg[cat]['info:data_dir'],
               cfg[cat].get(f"{cat}:hyperparams_csv_file", default_csv[cat]))
        df = pd.read_csv(file, header=0, sep=',')
        df = df.sort_values(['inference_period', 'domain', 'learning_space', 'experiment_serial'],
                             ascending=[True, True, True, True])
        df.to_csv(file, index=False, float_format='%.9g', header=True, mode='w')

#----------------
# Core functions
#----------------
@timeit
def construct_models(inference_prefix, domain_id, specs={}, desc=""):
    """
    Construct various kriging and GP models, with and without sequential or CRF simulations

    :param inference_prefix: (int) modelling period as described in `process`
    :param domain_id: (int) geologicial domain classification as described in `process`
    :param specs: (dict) configuration override dictionary
    :param desc: (str) provides a specific context for timeit feedback
    """
    mA = inference_prefix
    mA_mB_mC = '%02d_%02d_%02d' % (mA, mA+1, mA+2)
    model_names = []
    model_exec_times = []
    learn_times = []
    inference_times = []
    specs.update({'mA': mA, 'domain_id': domain_id})

    # Configure paths
    specs, data_path, subdir, result_path, figure_path = \
        configure_paths_and_conventions(inference_prefix, domain_id, specs)

    # Acquire data
    df_bh = pd.read_csv(f"{data_path}/blastholes_tagged.csv")
    df_bh = df_bh.rename(columns = {'EAST': 'X', 'NORTH': 'Y', 'RL': 'Z', 'PL_CU': 'V'})
    df_domain_x = pd.DataFrame(columns=['X','Y','Z','V']) #not using exploration assays
    df_domain_bh = df_bh[(df_bh['lookup_domain'] == domain_id) & np.isfinite(df_bh['V'].values)]
    min_v, max_v = np.min(df_domain_bh['V'].values), np.max(df_domain_bh['V'].values)
    R, S = compute_any_rotation_and_scaling_matrix(specs, domain_id)
    if subdir == 'learning_rotated':
        print(f"Rotation matrix R={R}\nScaling vector S={S}")

    # Visualise blastholes grades
    create_scatter_plot(
        df_domain_bh['X'], df_domain_bh['Y'], df_domain_bh['V'],
        min_v, max_v, symbsiz=50, palette='YlOrRd', cbtitle="Cu grade", symbol='s',
        graphtitle=f"Blastholes Cu grade - known samples for {mA_mB_mC}",
        savefile=os.path.join(figure_path, f"blastholes_grade_{domain_id}.pdf"))

    # Inference locations
    x_star_file = f"{data_path}/blocks_to_estimate_tagged.csv"
    if specs.get('inference_type', 'future-bench-prediction') == 'in-situ-regression':
        x_star_file = f"{data_path}/blocks_insitu_tagged.csv"

    df_k = pd.read_csv(x_star_file)
    df_domain_infer = pd.DataFrame(df_k[df_k['domain'] == domain_id])
    ground_truth = df_domain_infer['cu_bh_nn'].values

    # Configure Kriging and simulation parameters
    cfg_krige = create_kriging_config(R, S, specs)
    print(f"Processing {mA_mB_mC}, domain {domain_id}")
    print(f"- Training uses {len(df_domain_bh)} blastholes")
    print(f"- Inferencing: {len(df_domain_infer)} blocks")
    if cfg_krige['simulation:num'] > 256:
        specs['simulation:num'] = cfg_krige['simulation:num'] = 256
        print('The maximum number of simulations is capped at 256')

    #-----------------------------------
    # Technique 1
    # OK Sequential Gaussian Simulation
    #-----------------------------------
    print('\nTechnique 1: OK Sequential Gaussian Simulation\n' )
    cfg_oksgs = copy.deepcopy(cfg_krige)
    oksgs_abbrev = ''.join([x[0] for x in cfg_oksgs['kriging:type'].split('_')]) + 'sgs'
    oksgs_abbrev += 'r' if cfg_oksgs['kriging:transform_data'] else ''
    oksgs_csv = f"{result_path}/predictions-{oksgs_abbrev}-{domain_id}.csv"

    model_names.append('OK-SGS')
    if os.path.isfile(oksgs_csv):
        print(f"Reading {oksgs_csv}")
        cfg_oksgs['bypass_simulation'] = True   # only retrieve meta-data such as column labels
        KrigingManager.kriging_sequential_simulations(df_domain_bh, df_domain_infer, cfg_oksgs)
        df_oksgs = pd.read_csv(oksgs_csv, index_col=0, header=0)
        model_exec_times.append(0)
        learn_times.append(0)
        inference_times.append(0)
    else:
        print(f"Running simulation...\nResults will be saved to {oksgs_csv}")
        t0 = time.time()
        with warnings.catch_warnings():
            # Suppress the "DataFrame is highly fragmented" warning.
            # - This is usually the result of calling `frame.insert` many times,
            #   which has poor performance. Consider joining all columns at once
            #   using pd.concat(axis=1) instead.
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            df_oksgs = KrigingManager.kriging_sequential_simulations(df_domain_bh, df_domain_infer, cfg_oksgs)
        model_exec_times.append(time.time() - t0)
        learn_times.append(cfg_oksgs['timing:learn'])
        inference_times.append(cfg_oksgs['timing:inference'])
        print(f"Writing {oksgs_csv}")
        df_oksgs.to_csv(oksgs_csv)

    # OK-SGS: Compute mean, stdev using first m simulations
    max_simul = cfg_krige['simulation:num']
    two_powers = [2**i for i in np.arange(1, int(np.floor(np.log2(max_simul))+1))]
    oksgs_mean = {}
    oksgs_stdev = {}
    oksgs_sim_cols = [x for x in df_oksgs.columns]
    oksgs_vals = df_oksgs[oksgs_sim_cols].values
    for i in two_powers:
        oksgs_mean[f'from_{i}'] = np.mean(oksgs_vals[:,:i], axis=1)
        oksgs_stdev[f'from_{i}'] = np.std(oksgs_vals[:,:i], axis=1)

    # Check dictionary contents following simulations, including random states
    print('{}\n'.format(cfg_oksgs.keys()))
    print(f"simulation:period_domain_id: {cfg_oksgs['simulation:period_domain_id']}")
    print(f"simulation:period_domain_initial_state: {cfg_oksgs['simulation:period_domain_initial_state']}")
    print(f"simulation:path_seeds: {cfg_oksgs['simulation:path_seeds']}")

    # Check variogram parameters
    variogram_props = ['nugget','range','sill','nu','variogram-model','R','S']
    print("kriging:transform_data: {}".format(cfg_oksgs['kriging:transform_data']))
    print("variogram:params: {}".format(dict(zip(variogram_props, cfg_oksgs['variogram:params']))))

    # If training data was downsampled, selection will be fixed for subsequent models
    cfg_krige['data:downsampling_mask'] = cfg_oksgs.get('data:downsampling_mask', None)

    #--------------------------------------------------
    # Technique 2
    # Ordinary Kriging (without sequential simulation)
    # - Case 'nst': with normal score transformation
    # - Case 'raw': without data transformation
    #--------------------------------------------------
    print('\nTechnique 2: Ordinary Kriging\n' )
    cfg_ok = copy.deepcopy(cfg_krige)
    cfg_ok['kriging:type'] = 'ordinary_kriging'
    ok_mean, ok_var = {}, {}
    ok_col_mean, ok_col_stdev = {}, {}
    df_ok = pd.DataFrame()
    ok_abbrev = 'ok' + ('r' if cfg_ok['kriging:transform_data'] else '')
    ok_csv = f"{result_path}/predictions-{ok_abbrev}-{domain_id}.csv"

    if os.path.isfile(ok_csv):
        df_ok = pd.read_csv(ok_csv, index_col=0, header=0)
        (ok_mean, ok_var, ok_col_mean, ok_col_stdev) = \
            extract_raw_nst_predictions(df_ok, return_variance=True)
        model_names += ['OK_raw', 'OK_nst']
        model_exec_times += [0,0]
        learn_times += [0,0]
        inference_times += [0,0]
    else:
        for xform, bval in [('raw', False), ('nst', True)]:
            cfg_ok['kriging:apply_normal_score_transform'] = bval
            t0 = time.time()
            ok_mean[xform], ok_var[xform] = \
                KrigingManager.kriging_regression(df_domain_bh, df_domain_infer, cfg_ok)
            model_names.append(f"OK_{xform}")
            model_exec_times.append(time.time() - t0)
            learn_times.append(cfg_ok['timing:learn'])
            inference_times.append(cfg_ok['timing:inference'])
            ok_col_mean[xform] = 'ok_mean' + ('_nst' if xform == 'nst' else '')
            ok_col_stdev[xform] = 'ok_stdev' + ('_nst' if xform == 'nst' else '')
            df_ok[ok_col_mean[xform]] = ok_mean[xform]
            df_ok[ok_col_stdev[xform]] = np.sqrt(np.maximum(ok_var[xform], 0))

        df_ok = df_ok.set_index(df_domain_infer.index) #pd>Index generally not contiguous
        df_ok.to_csv(ok_csv)
        print(f"Writing {ok_csv}")

    for xform in ['raw', 'nst']:
        print('OK MSE[{}]: {}'.format(xform, np.mean((ok_mean[xform] - ground_truth)**2)))

    #-----------------------------------
    # Technique 3
    # SK Sequential Gaussian Simulation
    #-----------------------------------
    print('\nTechnique 3: SK Sequential Gaussian Simulation\n' )
    cfg_krige['kriging:type'] = 'simple_kriging'
    cfg_sksgs = copy.deepcopy(cfg_krige)
    cfg_sksgs['bypass_simulation'] = False
    sksgs_abbrev = ''.join([x[0] for x in cfg_sksgs['kriging:type'].split('_')]) + 'sgs'
    sksgs_abbrev += 'r' if cfg_sksgs['kriging:transform_data'] else ''
    sksgs_csv = f"{result_path}/predictions-{sksgs_abbrev}-{domain_id}.csv"

    model_names.append('SK-SGS')
    if os.path.isfile(sksgs_csv):
        print(f"Reading {sksgs_csv}")
        cfg_sksgs['bypass_simulation'] = True
        KrigingManager.kriging_sequential_simulations(df_domain_bh, df_domain_infer, cfg_sksgs)
        df_sksgs = pd.read_csv(sksgs_csv, index_col=0, header=0)
        model_exec_times.append(0)
        learn_times.append(0)
        inference_times.append(0)
    else:
        print(f"Running simulation...\nResults will be saved to {sksgs_csv}")
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            df_sksgs = KrigingManager.kriging_sequential_simulations(df_domain_bh, df_domain_infer, cfg_sksgs)
        model_exec_times.append(time.time() - t0)
        learn_times.append(cfg_sksgs['timing:learn'])
        inference_times.append(cfg_sksgs['timing:inference'])
        print(f"Writing {sksgs_csv}")
        df_sksgs.to_csv(sksgs_csv)

    sksgs_mean = {}
    sksgs_stdev = {}
    sksgs_sim_cols = [x for x in df_sksgs.columns]
    sksgs_vals = df_sksgs[sksgs_sim_cols].values
    for i in two_powers:
        sksgs_mean[f'from_{i}'] = np.mean(sksgs_vals[:,:i], axis=1)
        sksgs_stdev[f'from_{i}'] = np.std(sksgs_vals[:,:i], axis=1)

    print("kriging:transform_data: {}".format(cfg_sksgs['kriging:transform_data']))
    print("variogram:params: {}".format(dict(zip(variogram_props, cfg_sksgs['variogram:params']))))

    #--------------------------------------------------
    # Technique 4
    # Simple Kriging (without sequential simulation)
    #--------------------------------------------------
    print('\nTechnique 4: Simple Kriging\n' )
    cfg_sk = copy.deepcopy(cfg_krige)
    cfg_sk['kriging:type'] = 'simple_kriging'
    sk_mean, sk_var = {}, {}
    sk_col_mean, sk_col_stdev = {}, {}
    df_sk = pd.DataFrame()
    sk_abbrev = 'sk' + ('r' if cfg_sk['kriging:transform_data'] else '')
    sk_csv = f"{result_path}/predictions-{sk_abbrev}-{domain_id}.csv"

    if os.path.isfile(sk_csv):
        df_sk = pd.read_csv(sk_csv, index_col=0, header=0)
        (sk_mean, sk_var, sk_col_mean, sk_col_stdev) = \
            extract_raw_nst_predictions(df_sk, return_variance=True)
        model_names += ['SK_raw', 'SK_nst']
        model_exec_times += [0,0]
        learn_times += [0,0]
        inference_times += [0,0]
    else:
        for xform, bval in [('raw', False), ('nst', True)]:
            cfg_sk['kriging:apply_normal_score_transform'] = bval
            t0 = time.time()
            sk_mean[xform], sk_var[xform] = \
                KrigingManager.kriging_regression(df_domain_bh, df_domain_infer, cfg_sk)
            model_names.append(f"SK_{xform}")
            model_exec_times.append(time.time() - t0)
            learn_times.append(cfg_sk['timing:learn'])
            inference_times.append(cfg_sk['timing:inference'])
            sk_col_mean[xform] = 'sk_mean' + ('_nst' if xform == 'nst' else '')
            sk_col_stdev[xform] = 'sk_stdev' + ('_nst' if xform == 'nst' else '')
            df_sk[sk_col_mean[xform]] = sk_mean[xform]
            df_sk[sk_col_stdev[xform]] = np.sqrt(np.maximum(sk_var[xform], 0))

        df_sk = df_sk.set_index(df_domain_infer.index)
        df_sk.to_csv(sk_csv)
        print(f"Writing {sk_csv}")

    for xform in ['raw', 'nst']:
        print('SK MSE[{}]: {}'.format(xform, np.mean((sk_mean[xform] - ground_truth)**2)))

    # Configure Gaussian Process and simulation parameters
    # - Refer to doc string in `GPManager.gaussian_process_simulations`
    cfg_gp = create_gaussian_process_config(specs)

    # - For consistency, we adhere to the same choices if the
    #   training data has previously been downsampled.
    downsampling_mask = cfg_krige.get('data:downsampling_mask', None)
    if downsampling_mask is not None:
        cfg_gp['data:downsampling_mask'] = {
            'blasthole': downsampling_mask,
            'exploration': np.array([], dtype=int) #not used
        }

    #--------------------------------------------------
    # Technique 5
    # Gaussian Process Regression (Local Mean Approach)
    #--------------------------------------------------
    print('\nTechnique 5: Gaussian Process Regression (Local Mean Approach)\n' )
    method = 'GPR(L)'
    gpl_mean, gpl_stdev = {}, {}
    gpl_col_mean, gpl_col_stdev = {}, {}
    df_gpl = pd.DataFrame()
    gpl_abbrev = 'gpl' + ('r' if cfg_gp['gp:learning_inference_in_rotated_space'] else '')
    gpl_csv = f"{result_path}/predictions-{gpl_abbrev}-{domain_id}.csv"

    if os.path.isfile(gpl_csv):
        print(f"Reading {gpl_csv}")
        df_gpl = pd.read_csv(gpl_csv, index_col=0, header=0)
        (gpl_mean, gpl_stdev, gpl_col_mean, gpl_col_stdev) = \
            extract_raw_nst_predictions(df_gpl, return_variance=False)
        model_names += ['GP(L)_raw', 'GP(L)_nst']
        model_exec_times += [0,0]
        learn_times += [0,0]
        inference_times += [0,0]
    else:
        for xform, bval in [('raw', False), ('nst', True)]:
            cfg_gp['gp:apply_normal_score_transform'] = bval
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                t0 = time.time()
                # perform GPR
                gpl_mean[xform], gpl_stdev[xform] = \
                    GPManager.gaussian_process_simulations(
                    method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
                model_names.append(f"GP(L)_{xform}")
                model_exec_times.append(time.time() - t0)
                learn_times.append(cfg_gp['timing:learn'])
                inference_times.append(cfg_gp['timing:inference'])
                gpl_col_mean[xform] = cfg_gp['gp:mean_col_name']
                gpl_col_stdev[xform] = cfg_gp['gp:stdev_col_name']
                df_gpl[gpl_col_mean[xform]] = gpl_mean[xform]
                df_gpl[gpl_col_stdev[xform]] = gpl_stdev[xform]

        df_gpl = df_gpl.set_index(df_domain_infer.index)
        df_gpl.to_csv(gpl_csv)
        print(f"Writing {gpl_csv}")

    for xform in ['raw', 'nst']:
        print('GPR(L) MSE[{}]: {}'.format(xform, np.mean((gpl_mean[xform] - ground_truth)**2)))

    #-----------------------------------
    # Technique 6
    # GP Sequential Gaussian Simulation
    #-----------------------------------
    print('\nTechnique 6: GP Sequential Gaussian Simulation\n' )
    method = 'GP-SGS'
    cfg_gp['simulation:bypass'] = False
    gpsgs_abbrev = 'gpsgs' + ('r' if cfg_gp['gp:learning_inference_in_rotated_space'] else '')
    gpsgs_csv = f"{result_path}/predictions-{gpsgs_abbrev}-{domain_id}.csv"

    model_names.append('GP-SGS')
    if os.path.isfile(gpsgs_csv):
        print(f"Reading {gpsgs_csv}")
        df_gplsgs = pd.read_csv(gpsgs_csv, index_col=0, header=0)
        cfg_gp['simulation:column_names'] = list(df_gplsgs.columns)
        cfg_gp['simulation:bypass'] = True
        GPManager.gaussian_process_simulations(
            method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
        model_exec_times.append(0)
        learn_times.append(0)
        inference_times.append(0)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            print(f"Running simulation...\nResults will be saved to {gpsgs_csv}")
            t0 = time.time()
            df_gplsgs = GPManager.gaussian_process_simulations(
                       method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
            model_exec_times.append(time.time() - t0)
            learn_times.append(cfg_gp['timing:learn'])
            inference_times.append(cfg_gp['timing:inference'])
            print(f"Writing {gpsgs_csv}")
            df_gplsgs.to_csv(gpsgs_csv)

    gpsgs_mean = {}
    gpsgs_stdev = {}
    gpsgs_vals = df_gplsgs[cfg_gp['simulation:column_names']].values
    for i in two_powers:
        gpsgs_mean[f'from_{i}'] = np.mean(gpsgs_vals[:,:i], axis=1)
        gpsgs_stdev[f'from_{i}'] = np.std(gpsgs_vals[:,:i], axis=1)

    min_vsd = min([min(s) for s in gpsgs_stdev.values()])
    max_vsd = max([max(s) for s in gpsgs_stdev.values()])

    # Check dictionary contents following simulations, including random states
    print('{}\n'.format(cfg_gp.keys()))
    print(f"simulation:period_domain_id: {cfg_gp['simulation:period_domain_id']}")
    print(f"simulation:period_domain_initial_state: {cfg_gp['simulation:period_domain_initial_state']}")
    print(f"simulation:path_seeds: {cfg_gp['simulation:path_seeds']}")

    #--------------------------------------------------
    # Technique 7
    # Gaussian Process Regression (Global Mean Approach)
    #--------------------------------------------------
    print('\nTechnique 7: Gaussian Process Regression (Global Mean Approach)\n' )
    method = 'GPR(G)'
    gpg_mean, gpg_stdev = {}, {}
    gpg_col_mean, gpg_col_stdev = {}, {}
    df_gpg = pd.DataFrame()
    gpg_abbrev = 'gpg' + ('r' if cfg_gp['gp:learning_inference_in_rotated_space'] else '')
    gpg_csv = f"{result_path}/predictions-{gpg_abbrev}-{domain_id}.csv"

    if os.path.isfile(gpg_csv):
        print(f"Reading {gpg_csv}")
        df_gpg = pd.read_csv(gpg_csv, index_col=0, header=0)
        (gpg_mean, gpg_stdev, gpg_col_mean, gpg_col_stdev) = \
            extract_raw_nst_predictions(df_gpg, return_variance=False)
        model_names += ['GP(G)_raw', 'GP(G)_nst']
        model_exec_times += [0,0]
        learn_times += [0,0]
        inference_times += [0,0]
    else:
        for xform, bval in [('raw', False), ('nst', True)]:
            cfg_gp['gp:apply_normal_score_transform'] = bval
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                t0 = time.time()
                # perform GPR
                gpg_mean[xform], gpg_stdev[xform] = \
                    GPManager.gaussian_process_simulations(
                    method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
                model_names.append(f"GP(G)_{xform}")
                model_exec_times.append(time.time() - t0)
                learn_times.append(cfg_gp['timing:learn'])
                inference_times.append(cfg_gp['timing:inference'])
                gpg_col_mean[xform] = cfg_gp['gp:mean_col_name']
                gpg_col_stdev[xform] = cfg_gp['gp:stdev_col_name']
                df_gpg[gpg_col_mean[xform]] = gpg_mean[xform]
                df_gpg[gpg_col_stdev[xform]] = gpg_stdev[xform]

        df_gpg = df_gpg.set_index(df_domain_infer.index)
        df_gpg.to_csv(gpg_csv)
        print(f"Writing {gpg_csv}")

    for xform in ['raw', 'nst']:
        print('GPR(G) MSE[{}]: {}'.format(xform, np.mean((gpg_mean[xform] - ground_truth)**2)))

    #--------------------------------------
    # Technique 8
    # GP Spatially Correlated Random Field
    #--------------------------------------
    print('\nTechnique 8: GP Spatially Correlated Random Field\n' )
    method = 'GP-CRF'
    gpcrf_abbrev = 'gpcrf' + ('r' if cfg_gp['gp:learning_inference_in_rotated_space'] else '')
    gpcrf_csv = f"{result_path}/predictions-{gpcrf_abbrev}-{domain_id}.csv"

    model_names.append('GP-CRF')
    if os.path.isfile(gpcrf_csv):
        print(f"Reading {gpcrf_csv}")
        df_gplcrf = pd.read_csv(gpcrf_csv, index_col=0, header=0)
        cfg_gp['simulation:column_names'] = list(df_gplcrf.columns)
        cfg_gp['simulation:bypass'] = True
        GPManager.gaussian_process_simulations(method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
        model_exec_times.append(0)
        learn_times.append(0)
        inference_times.append(0)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            print(f"Running simulation...\nResults will be saved to {gpcrf_csv}")
            t0 = time.time()
            df_gplcrf = GPManager.gaussian_process_simulations(
                       method, df_domain_bh, df_domain_x, df_domain_infer, cfg_gp)
            model_exec_times.append(time.time() - t0)
            learn_times.append(cfg_gp['timing:learn'])
            inference_times.append(cfg_gp['timing:inference'])
            print(f"Writing {gpcrf_csv}")
            df_gplcrf.to_csv(gpcrf_csv)

    gpcrf_mean = {}
    gpcrf_stdev = {}
    gpcrf_vals = df_gplcrf[cfg_gp['simulation:column_names']].values
    for i in two_powers:
        gpcrf_mean[f'from_{i}'] = np.mean(gpcrf_vals[:,:i], axis=1)
        gpcrf_stdev[f'from_{i}'] = np.std(gpcrf_vals[:,:i], axis=1)

    # Record execution times
    num = len(model_names)
    df_t = pd.DataFrame({'inference_period': [inference_prefix] * num,
                         'domain_id': [domain_id] * num,
                         'training_samples': [len(df_domain_bh)] * num,
                         'inference_locations': [len(df_domain_infer)] * num,
                         'model': model_names,
                         'execution_time': model_exec_times,
                         'learn_time': learn_times,
                         'inference_time': inference_times})
    timing_csv = f"{result_path}/timing_{domain_id}.csv"
    if not os.path.exists(timing_csv):
        df_t.to_csv(timing_csv)

    # Assemble predictions for all candidate models
    candidates_mu = OrderedDict(
                    [('SK', sk_mean['raw']), ('SK_nst', sk_mean['nst']),
                     ('OK', ok_mean['raw']), ('OK_nst', ok_mean['nst']),
                     ('GP(L)', df_gpl[gpl_col_mean['raw']].values),
                     ('GP(L)_nst', df_gpl[gpl_col_mean['nst']].values),
                     ('GP(G)', df_gpg[gpg_col_mean['raw']].values),
                     ('GP(G)_nst', df_gpg[gpg_col_mean['nst']].values)
                    ])
    candidates_sigma = OrderedDict(
                       [('SK', np.sqrt(sk_var['raw'])), ('SK_nst', np.sqrt(sk_var['nst'])),
                        ('OK', np.sqrt(ok_var['raw'])), ('OK_nst', np.sqrt(ok_var['nst'])),
                        ('GP(L)', df_gpl[gpl_col_stdev['raw']].values),
                        ('GP(L)_nst', df_gpl[gpl_col_stdev['nst']].values),
                        ('GP(G)', df_gpg[gpg_col_stdev['raw']].values),
                        ('GP(G)_nst', df_gpg[gpg_col_stdev['nst']].values)
                       ])
    for i in two_powers:
        key = f"from_{i}"
        candidates_mu[f"SK_SGS_{key}"] = sksgs_mean[key]
        candidates_sigma[f"SK_SGS_{key}"] = sksgs_stdev[key]
    for i in two_powers:
        key = f"from_{i}"
        candidates_mu[f"OK_SGS_{key}"] = oksgs_mean[key]
        candidates_sigma[f"OK_SGS_{key}"] = oksgs_stdev[key]
    for i in two_powers:
        key = f"from_{i}"
        candidates_mu[f"GP_SGS_{key}"] = gpsgs_mean[key]
        candidates_sigma[f"GP_SGS_{key}"] = gpsgs_stdev[key]
    for i in two_powers:
        key = f"from_{i}"
        candidates_mu[f"GP_CRF_{key}"] = gpcrf_mean[key]
        candidates_sigma[f"GP_CRF_{key}"] = gpcrf_stdev[key]

    # Write results to pickle file
    for k in ['timing:learn', 'timing:inference']:
        _ = cfg_krige.pop(k, None)
        _ = cfg_gp.pop(k, None)

    models_pfile = f"{result_path}/models-{domain_id}.p"
    with open(models_pfile, 'wb') as hdl:
        variables = [candidates_mu, candidates_sigma, cfg_krige, cfg_gp]
        pickle.dump(variables, hdl, protocol=4)
    
@timeit
def analyse_models(inference_prefix, domain_id, specs={}, desc=""):
    """
    Compute histograms, variograms and uncertainty-based prediction performance statistics
    """
    # Configure paths
    specs, data_path, subdir, result_path, figure_path = \
        configure_paths_and_conventions(inference_prefix, domain_id, specs)

    models_pfile = f"{result_path}/models-{domain_id}.p"
    histogram_stats_pfile = f"{result_path}/histogram-stats-{domain_id}.p"
    variogram_stats_pfile = f"{result_path}/variogram-stats-{domain_id}.p"
    uncertainty_stats_pfile = f"{result_path}/uncertainty-stats-{domain_id}.p"
    analysis_csv = f"{result_path}/analysis-{domain_id}.csv"
    hdist_crossplots_file = f"{figure_path}/histogram-dist-crossplots-{domain_id}.pdf"
    histogram_file = f"{figure_path}/histograms-{domain_id}.pdf"
    variogram_file = f"{figure_path}/variograms-@-{domain_id}.pdf"
    kappa_w_file = f"{figure_path}/uncertainty_kappa-{domain_id}.pdf"
    spatial_mean_file = f"{figure_path}/spatial_mean-@-{domain_id}.pdf"
    spatial_stdev_file = f"{figure_path}/spatial_stdev-@-{domain_id}.pdf"
    signed_distortion_file = f"{figure_path}/uncertainty_signed_distortion-{domain_id}.pdf"
    likelihood_file = f"{figure_path}/uncertainty_likelihood-{domain_id}.pdf"
    x_star_file = f"{data_path}/blocks_to_estimate_tagged.csv"
    if specs.get('inference_type', 'future-bench-prediction') == 'in-situ-regression':
        x_star_file = f"{data_path}/blocks_insitu_tagged.csv"

    if not os.path.exists(models_pfile):
        raise RuntimeError(f"Required file {models_pfile} is missing. You may need to "
                            "run `construct_models` to generate prediction results")
    with open(models_pfile, 'rb') as f:
        (candidates_mu, candidates_sigma, cfg_krige, cfg_gp) = pickle.load(f)

    df_bh = pd.read_csv(f"{data_path}/blastholes_tagged.csv")
    df_bh = df_bh.rename(columns = {'EAST': 'X', 'NORTH': 'Y', 'RL': 'Z', 'PL_CU': 'V'})
    df_domain_bh = df_bh[(df_bh['lookup_domain'] == domain_id) & np.isfinite(df_bh['V'].values)]
    df_k = pd.read_csv(x_star_file)
    df_domain_infer = pd.DataFrame(df_k[df_k['domain'] == domain_id])
    mu_0 = ground_truth = df_domain_infer['cu_bh_nn'].values
    max_simul = cfg_krige['simulation:num']
    min_v, max_v = np.min(df_domain_bh['V'].values), np.max(df_domain_bh['V'].values)
    two_powers = [int(k.split('_')[-1]) for k in candidates_mu if 'CRF_from' in k]    
    min_vsd = min([min(v) for k,v in candidates_sigma.items() if 'GP_SGS_from' in k])
    max_vsd = max([max(v) for k,v in candidates_sigma.items() if 'GP_SGS_from' in k])

    #--------------------------------------
    # Category 1 (Global measures)
    # Histograms for model mean predictions
    #--------------------------------------
    if os.path.exists(histogram_stats_pfile):
        with open(histogram_stats_pfile, 'rb') as f:
            (candidates_psChi2, candidates_JS, candidates_IOU, candidates_EM,
             candidates_hist, candidates_RMSE, bins_representation, df_h) = pickle.load(f)
        stat_names = list(df_h.columns)
    else:
        candidates_psChi2 = OrderedDict()
        candidates_JS = OrderedDict()
        candidates_IOU = OrderedDict()
        candidates_EM = OrderedDict()
        candidates_hist = OrderedDict()
        candidates_RMSE = OrderedDict()

        for k in candidates_mu.keys():
            d = compute_histogram_statistics(mu_0, candidates_mu[k])
            candidates_psChi2[k] = d['psym-chi-square']
            candidates_JS[k] = d['jensen-shannon']
            candidates_IOU[k] = d['ruzicka']
            candidates_EM[k] = d['wasserstein']
            candidates_hist[k] = d['pmf_x']
            candidates_RMSE[k] = compute_root_mean_squared_error(mu_0, candidates_mu[k])

        bins_representation = d['values']
        candidates_hist['GroundTruth'] = d['pmf_y']

        # Rank models by histogram distances
        compute_rank = lambda seq: np.array([seq[k] for k in candidates_mu.keys()]).argsort().argsort() + 1
        rank_psChi2 = compute_rank(candidates_psChi2)
        rank_JS = compute_rank(candidates_JS)
        rank_IOU = compute_rank(candidates_IOU)
        rank_EM = compute_rank(candidates_EM)
        rank_mean = np.mean(np.c_[rank_psChi2, rank_JS, rank_IOU, rank_EM], axis=1)
        rank_overall = rank_mean.argsort().argsort() + 1

        data = OrderedDict()
        stat_names = ['d(psChi2)', 'd(JS)', 'd(IOU)', 'd(EM)']
        for k in candidates_mu.keys():
            data[k] = np.round([candidates_psChi2[k], candidates_JS[k], candidates_IOU[k], candidates_EM[k]], 4)
        df_h = pd.DataFrame.from_dict(data, orient='index', columns=stat_names)
        df_h['rank(psChi2)'] = rank_psChi2
        df_h['rank(JS)'] = rank_JS
        df_h['rank(IOU)'] = rank_IOU
        df_h['rank(EM)'] = rank_EM
        df_h['rank(avg)'] = rank_mean
        df_h['rank(overall)'] = rank_overall

        with open(histogram_stats_pfile, 'wb') as hdl:
            variables = [candidates_psChi2, candidates_JS, candidates_IOU, candidates_EM,
                         candidates_hist, candidates_RMSE, bins_representation, df_h]
            pickle.dump(variables, hdl, protocol=4)

    # Examine cross-plots of histogram measures to check consistency
    fig = plt.figure(figsize=(10,8)) #(x,y)
    array_ = lambda d: np.array([d[k] for k in d.keys() if k != 'SK'])
    data = [array_(candidates_psChi2), array_(candidates_JS),
            array_(candidates_IOU), array_(candidates_EM)]
    stat_full_names = [r'Prob.Symm.$\chi^2$', 'Jensen-Shannon', 'Ruzicka.IOU', 'Wasserstein.EM']
    stat_abbrevs = [x.strip('d()') for x in stat_names]
    pairs = [(0,[1,2,3]), (1,[2,3]), (2,[3])]
    for row, columns in pairs:
        for col in columns:
            plt.subplot(3,3,row*3+col)
            plt.plot([min(data[col]),max(data[col])], [min(data[row]),max(data[row])], c="#cccccc")
            plt.scatter(data[col], data[row], s=10)
            if col == columns[0]:
                plt.xlabel(stat_abbrevs[col], fontsize=8)
                plt.ylabel(stat_abbrevs[row], fontsize=8)
            else:
                plt.xticks([])
            pearson = np.corrcoef(data[row], data[col])[0,1]
            plt.text(min(data[col]), 0.6*max(data[row]), r"$\rho$={}".format('%.4f' % pearson))
            plt.title(f"{stat_full_names[row]} vs {stat_full_names[col]}", fontsize=9)
    plt.savefig(hdist_crossplots_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    # Draw histograms for selected models
    nsim = min(32, 2**int(np.floor(np.log2(max_simul))))
    selected = ['SK_nst', 'OK_nst', 'GP(G)_nst', 'GP(L)_nst',
                f"SK_SGS_from_{nsim}", f"OK_SGS_from_{nsim}",
                f"GP_CRF_from_{nsim}", f"GP_SGS_from_{nsim}"]
    fig = plt.figure(figsize=(12,15))
    w = np.median(np.diff(bins_representation))
    for i, k in enumerate(selected):
        p = candidates_hist[k]
        q = candidates_hist['GroundTruth']
        plt.subplot(4,2,i+1)
        plt.bar(bins_representation, p, edgecolor=None, width=w, label=k)
        plt.bar(bins_representation, q, facecolor=None, fill=False, edgecolor='k', width=w, label='GroundTruth')
        plt.legend(fontsize=9, loc='upper left')
        if i >= len(selected)-2:
            plt.xlabel('grade')
        plt.title(f"{k} probability mass function", fontsize=10)
    plt.savefig(histogram_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    #----------------------------------------
    # Category 2 (Local correlation measures)
    # Variograms for model mean predictions
    #----------------------------------------
    cfg_krige['variogram:use_nugget'] = specs.get('variogram:use_nugget', False)
    cfg_krige['variogram:max_lag'] = specs.get('variogram:max_lag', 250.0)
    cfg_krige['variogram:num_lag'] = specs.get('variogram:num_lag', 45)
    cfg_krige['variogram:required_samples'] = specs.get('variogram:required_samples', 30)
    vgram = {}
    variogram_model = cfg_krige.get('kriging:covariance_fn', 'matern')
    fixed_nu = cfg_krige.get('kriging:matern_smoothness', None)
    max_range = 2 * cfg_krige['variogram:max_lag']

    # Option to constrain the `nu` smoothness parameter for Matern.
    # General constraints = (lower, upper) where for instance
    # upper = [max_range, max_sill, max_nu, max_nugget]
    constraints = None
    if fixed_nu:
        constraints = ([0., 0., fixed_nu - 0.0001, 0.],
                       [max_range, max_var, fixed_nu + 0.0001, 0.5 * max_var])
        if cfg_krige['variogram:use_nugget'] is False:
            constraints = tuple([cs[:-1] for cs in constraints])

    make_variogram = lambda x, y : skg.Variogram(
                     coordinates=x,
                     values=y,
                     estimator='matheron',
                     model=variogram_model,
                     use_nugget=cfg_krige['variogram:use_nugget'],
                     maxlag=max_range,
                     n_lags=cfg_krige['variogram:num_lag'],
                     fit_bounds=constraints)

    class VariogramValues:
        def __init__(self, b=None, e=None):
            self.bins = b
            self.experimental = e

    # Retrieve precomputed values
    sgs_vals = dict()
    for model in ['SK_SGS', 'OK_SGS', 'GP_SGS', 'GP_CRF']:
        sgs_abbrev = model.replace('_', '').lower() \
                   + ('r' if cfg_krige['kriging:transform_data'] else '')
        sgs_csv = f"{result_path}/predictions-{sgs_abbrev}-{domain_id}.csv"
        sgs_vals[model] = pd.read_csv(sgs_csv, index_col=0, header=0).values

    # Compute variograms
    all_variograms_computed = False
    try:
        # Compute variograms for models in the focused group
        if len(ground_truth) >= cfg_krige['variogram:required_samples']:
            print(f"computing variograms", end=': ')
            # Compute variograms for candidates
            for model in candidates_mu.keys():
                x = df_domain_infer[['X','Y','Z']].values
                y = candidates_mu[model].flatten()
                retain = np.where(np.isfinite(y))[0]
                vgram[model] = make_variogram(x[retain], y[retain])
                print(f"{model}", end=', ')

            # Compute variograms for ground truth types
            for model, x, y in zip(['GroundTruth(blocks)', 'GroundTruth(training bh)'],
                                   [df_domain_infer[['X','Y','Z']].values, df_domain_bh[['X','Y','Z']].values],
                                   [ground_truth, df_domain_bh['V'].values]):
                retain = np.where(np.isfinite(y))[0]
                vgram[model] = make_variogram(x[retain], y[retain])
                print(f"{model}", end=', ')

            # Compute variogram stats for models declared thus far
            # - df_v.columns = ["method", "serial", "bins", "variogram($A)", "p25($A)", "p50($A)",
            #                    "p75($A)", ratios($A), "p25($B)", "p50($B)", "p75($B)", ratios($B)]
            #   where "$A"="GroundTruth(training bh)", "$B"="GroundTruth(training bh)"
            #         "method" is synonymous with "model" (such as "OK-SGS")
            #         "serial" describes the inference period and domain as f"{mA}:{domain_id}"
            ratios, stats, df_v = compute_variogram_stats(
                                  vgram,
                                  serial=f"{inference_prefix}:{domain_id}",
                                  references=['GroundTruth(blocks)', 'GroundTruth(training bh)'],
                                  percentiles=[25,50,75])

            # Compute typical variogram from single-shot simulation
            x = df_domain_infer[['X','Y','Z']].values
            for model in ['SK_SGS', 'OK_SGS', 'GP_SGS', 'GP_CRF']:
                individual_vgram = []
                pred = sgs_vals[model]
                for simul in np.arange(min(pred.shape[1], 16)):
                    y = pred[:, simul]
                    retain = np.where(np.isfinite(y))[0]
                    vario = make_variogram(x[retain], y[retain])
                    individual_vgram.append(vario.experimental)
                key = model + '_single'
                vgram[key] = VariogramValues(vario.bins, np.mean(individual_vgram, axis=0))
                print(f"{key}", end=', ')

            # Compute variograms for remaining "black box" models
            for model, x, y in zip(['RTK_LongRangeModel'],
                                   [df_domain_infer[['X','Y','Z']].values] * 1,
                                   [df_domain_infer['cu_50'].values]):
                retain = np.where(np.isfinite(y))[0]
                vgram[model] = make_variogram(x[retain], y[retain])
                print(f"{model}", end=', ')

            with open(variogram_stats_pfile, 'wb') as hdl:
                pickle.dump([ratios, stats, df_v], hdl, protocol=4)

            all_variograms_computed = True

    except AttributeError as e: #handle situation where all values are the same
        print(e)

    if not all_variograms_computed:
        models = list(candidates_mu.keys()) + ['GroundTruth(blocks)', 'GroundTruth(training bh)']
        vgram_ratios50 = [np.nan] * len(models)
        df_v = pd.DataFrame(zip(models, vgram_ratios50), columns=['method','p50(GroundTruth(blocks))'])

    # Generate variogram plots using a consistent colour scheme
    # Curves for model <m> consist of
    #   ["<m>_SGS_single"] +                      #R.1
    #   ["<m>_SGS_from{t}" for t in two_powers] + #R.2
    #   ["<m>_nst"] if <m> ! 'GP(L)' else [] +    #R.3
    #   ['GP(L)_nst] +                            #R.4
    #   ["RTK_LongRangeModel"] +                  #R.5
    #   ["GroundTruth(blocks)", "GroundTruth(training bh)"] #R.6
    two_powers = [int(k.split('_')[-1]) for k in candidates_mu if 'CRF_from' in k]
    reserved_patterns = ['-<','->','-x','-o','--*','-.p',':P',':s']
    n = len(two_powers)
    clut = lambda words: [variograms_colour_lookup(w) for w in words]
    common = ['lilac', 'gray'] + ['black']*2 #covers R.4-R.6
    palette = {'SK_SGS': clut(['olive'] + ['green-mid']*n + ['green-dark'] + common),
               'OK_SGS': clut(['magenta'] + ['blue-light']*n + ['blue-dark'] + common),
               'GP_SGS': clut(['red-dark'] + ['red']*n + common),
               'GP_CRF': clut(['orange-dark'] + ['orange']*n + ['pink'] + common)
              }

    if all_variograms_computed and len(ground_truth) >= cfg_krige['variogram:required_samples']:
        vgram_peak = max([np.nanmax(v.experimental) for k,v in vgram.items()])
        vgram_trough = min([np.nanmin(v.experimental) for k,v in vgram.items()])
        linear_vertical_scale = specs.get('variogram:linear_vertical_scale', False)
        category_items = OrderedDict()

        categories = ['SK_SGS', 'OK_SGS', 'GP_SGS', 'GP_CRF']
        base_techniques = ['SK_nst', 'OK_nst', 'GP(L)_nst', 'GP(G)_nst']
        for base, cat in zip(base_techniques, categories):
            print(f"Producing variogram plot for {cat}")
            category_items[cat] = [f"{cat}_single"] \
                                + [x for x in vgram.keys() if f"{cat}_from" in x] \
                                + [base] + (['GP(L)_nst'] if base != 'GP(L)_nst' else []) \
                                + ['RTK_LongRangeModel'] \
                                + ['GroundTruth(blocks)', 'GroundTruth(training bh)']
            patterns = ['-^'] + reserved_patterns[:n] \
                     + (['-o'] if base != 'GP' else []) + ['-o','-o','-o','-o','--o']
            colors = palette[cat]
            plt.figure(figsize=(10,8))
            for i, model in enumerate(category_items[cat]):
                xe = vgram[model].bins
                ye = vgram[model].experimental
                if linear_vertical_scale:
                    plt.plot(xe, ye, patterns[i], color=colors[i], markersize=4, label=model)
                    legend_position = 'upper left'
                else:
                    plt.semilogy(xe, ye, patterns[i], color=colors[i], markersize=4, label=model)
                    legend_position = 'lower right'
            plt.title(f"Variograms comparison (models in the {cat} family)")
            plt.ylim([vgram_trough, np.ceil(1000 * vgram_peak) / 1000])
            plt.xlabel('Lag [m]')
            plt.ylabel('Semivariance')
            plt.legend(loc=legend_position)
            plt.savefig(variogram_file.replace('@', cat.lower()), bbox_inches='tight', pad_inches=0.05)
            plt.clf()
            plt.close()

    #----------------------------------------------
    # Category 3 (Uncertainty-based measures)
    # Evaluate performance of probabilistic models
    #----------------------------------------------
    # Specify a default p vector with 256 bins
    # - comprising two linear segments with denser spacing in the upper echelon
    default_p_values = np.r_[np.linspace(0,0.98,247)[1:-1], np.linspace(0.98,1,12)[:-1]]
    p_values = specs.get('uncertainty:p_values', default_p_values)

    if os.path.exists(uncertainty_stats_pfile):
        with open(uncertainty_stats_pfile, 'rb') as f:
            (candidates_s, candidates_L, candidates_K, candidates_A,
             candidates_P, candidates_G, candidates_W, candidates_T,
             candidates_invalid, df_a, df_s) = pickle.load(f)
        stat_names = list(df_s.columns)
    else:
        # Investigate the sensitivity of kappa accuracy measure
        candidates_A_slack = OrderedDict()
        keys_a = []
        for xi in [0, 0.005, 0.01, 0.05, 0.1, 0.25]:
            keys_a.append(f"Accuracy({xi})")
            candidates_A_slack[xi] = OrderedDict()
            for k in candidates_mu.keys():
                s_scores = compute_model_likelihood(mu_0, candidates_mu[k], candidates_sigma[k])[1]
                accuracy = compute_distribution_accuracy(p_values, None, s_scores, slack=xi)
                candidates_A_slack[xi][k] = accuracy
     
        data_a = OrderedDict()
        for k in candidates_mu.keys():
            data_a[k] = np.round([d[k] for s, d in candidates_A_slack.items()], 4)
        df_a = pd.DataFrame.from_dict(data_a, orient='index', columns=keys_a)

        # Compute uncertainty-based measures
        # - s, L and K denote the signed scores, likelihood, and mean kappa proportions
        # - A, P, G, W, and T denote the accuracy, precision, goodness, width and tightness
        #   of the conditional distribution function from a given model
        candidates_s = OrderedDict()
        candidates_L = OrderedDict()
        candidates_K = OrderedDict()
        candidates_A = OrderedDict()
        candidates_P = OrderedDict()
        candidates_G = OrderedDict()
        candidates_W = OrderedDict()
        candidates_T = OrderedDict()
        candidates_invalid = OrderedDict()
        for k in candidates_mu.keys():
            d = compute_performance_statistics(mu_0, candidates_mu[k], candidates_sigma[k], slack=0.05)
            candidates_s[k] = d['s_scores']
            candidates_L[k] = np.mean(d['likelihood'])
            candidates_K[k] = d['proportion']
            candidates_A[k] = d['accuracy']
            candidates_P[k] = d['precision']
            candidates_G[k] = d['goodness']
            candidates_W[k] = d['width']
            candidates_T[k] = d['tightness']
            candidates_invalid[k] = d['invalid_samples']

        data = OrderedDict()
        stat_names = ['h(psChi2)', 'h(JS)', 'h(IOU)', 'h(EM)', 'h(rank)', 'RMSE',
                      'Variogram Ratios', 'Spatial Fidelity', '|s|_L', '|s|_U',
                      'Likelihood', 'Accuracy(.05)', 'Precision', 'Goodness', 'Tightness']
        for k in candidates_mu.keys():
            s = np.abs(candidates_s[k])
            variogram_ratio = df_v[df_v.method==k]['p50(GroundTruth(blocks))'].values[0]
            spatial_fidelity = np.sqrt(1 - np.abs(np.minimum(variogram_ratio, 2) - 1))
            data[k] = [candidates_psChi2[k], candidates_JS[k], candidates_IOU[k],
                       candidates_EM[k], df_h.loc[k,'rank(overall)'], candidates_RMSE[k],
                       variogram_ratio, spatial_fidelity, np.percentile(s,25), np.percentile(s,75),
                       candidates_L[k], candidates_A[k], candidates_P[k], candidates_G[k], candidates_T[k]]
        df_s = pd.DataFrame.from_dict(data, orient='index', columns=stat_names)
        df_s.to_csv(analysis_csv)

        with open(uncertainty_stats_pfile, 'wb') as hdl:
            variables = [candidates_s, candidates_L, candidates_K, candidates_A,
                         candidates_P, candidates_G, candidates_W, candidates_T,
                         candidates_invalid, df_a, df_s]
            pickle.dump(variables, hdl, protocol=4)

    # Produce graphs on prediction uncertainty accuracy and interval width
    p_vals = np.r_[np.linspace(0,1,41)[1:-1], 0.9825, 0.99, 0.997]
    alpha = np.linspace(0.025, 0.99, 100)
    n_sigma = norm.ppf(1 - (1 - np.array(alpha))/2, 0, 1)

    fig = plt.figure(figsize=(12,30))
    for i, k in enumerate(selected):
        s_scores = candidates_s[k]
        retain = np.isfinite(mu_0 + candidates_mu[k] + candidates_sigma[k])
        sigma_hat = candidates_sigma[k][retain]
        y_var = np.nanvar(ground_truth)
        signal_variance = y_var if y_var > 0 else 1.0
        kappa_scores = compute_kappa_bar(s_scores, p_vals)
        interval_widths = compute_width_bar(sigma_hat, s_scores, p_vals)
        # kappa (accuracy) plots
        plt.subplot(8,2,2*i+1)
        plt.plot(p_vals, p_vals, '-', color="#888888")
        plt.plot(p_vals, kappa_scores, 'k')
        plt.ylim([0,1])
        plt.text(0.375, 0.25, "Distribution accuracy: A=%.6f" % candidates_A[k], fontsize=9)
        plt.text(0.375, 0.175, "Coverage precision: P=%.6f" % candidates_P[k], fontsize=9)
        plt.text(0.375, 0.1, "Deutsch goodness statistic: G=%.6f" % candidates_G[k], fontsize=9)
        plt.legend(["expected proportions", r"$\kappa(\hat{\mu},\hat{\sigma}\mid\mu_0$)"], loc='upper left')
        plt.title(f"Kappa plot - Expected vs estimated proportions for {k}", fontsize=10)
        if i >= len(selected) - 1:
            plt.xlabel('p')
        # W plots
        plt.subplot(8,2,2*i+2)
        n_sigma = norm.ppf(1 - (1 - np.array(alpha))/2, 0, 1)
        plt.plot(alpha, n_sigma, '-', color="#888888")
        plt.plot(p_vals, interval_widths / np.sqrt(signal_variance), 'k')
        plt.ylim([0,2])
        plt.text(0.0, 1.4, "Black curve: the lower the better", fontsize=9, color="#888888")
        plt.text(0.25, 0.1, r"Tightness statistic: T=%.6f [for $\kappa_j(p)>p]$" % candidates_T[k], fontsize=9)
        plt.legend(["norminv(1-(1-p)/2)", r"Width of prediction interval: $\bar{W}(p)/\sigma_Y$"], loc='upper left')
        plt.title(f"W plot - normalised predict interval width vs p for {k}", fontsize=10)
        if i >= len(selected) - 1:
            plt.xlabel('p')    
    plt.savefig(kappa_w_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    # Visualisation of model estimates "mu" and "sigma"
    fig = plt.figure(figsize=(12,12))
    print("Predicted mean comparison for main model candidates")
    for i, k in enumerate(selected):
        cbar_desc = "mu(Cu)"
        hide_x_axis = False if i >= len(selected)-2 else True
        hide_y_axis = False if i%2==0 else True
        create_scatter_plot(
            df_domain_infer['X'], df_domain_infer['Y'], candidates_mu[k],
            min_v, max_v, symbsiz=50, subplotargs=[4,2,i+1], palette='YlOrRd', cbtitle=cbar_desc,
            sharex=hide_x_axis, sharey=hide_y_axis, symbol='s',
            graphtitle=f"{k} predicted mean")
    plt.savefig(spatial_mean_file.replace('@', 'for_candidates'), bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    print("Predicted mean as a function of number of simulations")
    for cat in ['OK_SGS', 'GP_SGS', 'GP_CRF']:
        fig = plt.figure(figsize=(12,12))
        for i, m in enumerate(two_powers):
            create_scatter_plot(
                df_domain_infer['X'], df_domain_infer['Y'], candidates_mu[f"{cat}_from_{m}"],
                min_v, max_v, symbsiz=50, subplotargs=[4,2,i+1], palette='YlOrRd', cbtitle="mu(Cu)",
                sharex=True, symbol='s', graphtitle=f"{cat} predicted Cu mean(from {m} runs)")
        plt.savefig(spatial_mean_file.replace('@', f"{cat}_convergence"), bbox_inches='tight', pad_inches=0.05)
        plt.clf()
        plt.close()

    fig = plt.figure(figsize=(12,12))
    print("Predicted standard deviation comparison for main model candidates")
    for i, k in enumerate([m for m in selected if m not in ['SK','OK']]): #code based on notebook 2D.7
        cbar_desc = "sigma(Cu)"
        hide_x_axis = False if i >= len(selected)-2 else True
        hide_y_axis = False if i%2==0 else True
        ymin = np.percentile(candidates_sigma[k], 5) if k in ['SK','OK'] else min_vsd
        ymax = np.percentile(candidates_sigma[k], 95) if k in ['SK','OK'] else max_vsd
        create_scatter_plot(
            df_domain_infer['X'], df_domain_infer['Y'], candidates_sigma[k],
            ymin, ymax, symbsiz=50, subplotargs=[4,2,i+1], palette='Blues', cbtitle=cbar_desc,
            sharex=hide_x_axis, sharey=hide_y_axis, symbol='s',
            graphtitle=f"{k} predicted stdev")
    plt.savefig(spatial_stdev_file.replace('@', 'for_candidates'), bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    print("Predicted standard deviation as a function of number of simulations")
    for cat in ['OK_SGS', 'GP_SGS', 'GP_CRF']:
        fig = plt.figure(figsize=(12,12))
        for i, m in enumerate(two_powers):
            create_scatter_plot(
                df_domain_infer['X'], df_domain_infer['Y'], candidates_sigma[f"{cat}_from_{m}"],
                min_vsd, max_vsd, symbsiz=50, subplotargs=[4,2,i+1], palette='Blues', cbtitle="sigma(Cu)",
                sharex=True, symbol='s', graphtitle=f"{cat} predicted Cu stdev(from {m} runs)")
        plt.savefig(spatial_stdev_file.replace('@', f"{cat}_convergence"), bbox_inches='tight', pad_inches=0.05)
        plt.clf()
        plt.close()

    # Visualise signed likelihood scores $s(\hat{\mu},\hat{\sigma}\mid\mu_0)$
    cmapBlRd = create_inverted_colormap(gamma=2.5)
    fig = plt.figure(figsize=(12,15))
    print('Using s score to illustrate relative distortion')
    print('Visual interpretation: Map of questionable or least probable predictions')
    print(' - Dark: bad, light: good')
    print(' - Red: under-estimated, blue: over-estimated')
    displayed_models = [m for m in selected if m!='OK' and m!='SK']
    total_samples = len(df_domain_infer['X'])
    for i, k in enumerate(displayed_models):
        cbar_desc = "red=under-estimated, 0=worst" if i%2==1 else "quality"
        hide_x_axis = False if i >= len(displayed_models)-2 else True
        hide_y_axis = False if i%2==0 else True
        retain = np.setdiff1d(range(total_samples), candidates_invalid[k])
        create_scatter_plot(
            df_domain_infer['X'].iloc[retain], df_domain_infer['Y'].iloc[retain],
            candidates_s[k], -1, +1, symbsiz=50, subplotargs=[4,2,i+1], palette=cmapBlRd,
            cbtitle=cbar_desc, sharex=hide_x_axis, sharey=hide_y_axis, symbol='s',
            graphtitle=r"Distortion Map, $s(\hat{\mu},\hat{\sigma}\mid\mu_0)$ for " + f"{k}")
    plt.savefig(signed_distortion_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

    cmapBl = create_inverted_colormap(gamma=0.7, monochrome=True)
    fig = plt.figure(figsize=(12,15))
    print('Using likelihood score to illustrate quality of probabilistic prediction')
    for i, k in enumerate(displayed_models):
        hide_x_axis = False if i >= len(displayed_models)-2 else True
        hide_y_axis = False if i%2==0 else True
        retain = np.setdiff1d(range(total_samples), candidates_invalid[k])
        create_scatter_plot(
            df_domain_infer['X'].iloc[retain], df_domain_infer['Y'].iloc[retain],
            np.abs(candidates_s[k]), 0, +1, symbsiz=50, subplotargs=[4,2,i+1], palette=cmapBl,
            cbtitle="likelihood", sharex=hide_x_axis, sharey=hide_y_axis, symbol='s',
            graphtitle=r"Uncertainty Likelihood $l(\hat{\mu},\hat{\sigma}\mid\mu_0)$ for " + f"{k}")
    plt.savefig(likelihood_file, bbox_inches='tight', pad_inches=0.05)
    plt.clf()
    plt.close()

@timeit
def process(inference_prefix, domain_id, num_simulations, **kwargs):
    """
    Coordinate model creation and evaluation process

    :param inference_prefix: (int) synonymous with symbol "mA", where the modelling
                                   period "mA_mB_mC" indicates an intent to estimate
                                   the grade for a 3 month period starting from month
                                   "mA" using previously gathered data.
    :param domain_id: (int) a four-digit geologicial domain classification (LGPR)
                            where L, G, P and R denote limb zone, grade zone, prophyry
                            zone and rock type interpretations, respectively.
    :param num_simulations: (int) number of simulation rounds for both sequential
                                  Gaussian, and correlated random field simulations.
    """
    learning_frame = check_learning_rotation_status(kwargs)
    learning_frame_prefix = 'r' if learning_frame == 'learning_rotated' else 'n'
    print(f"Processing mA={inference_prefix}, domain={domain_id}, {learning_frame} space")
    tag = f"{inference_prefix}:{domain_id}:{learning_frame_prefix}"
    # permit override of default parameters if the key is found within the options list
    options = list(create_kriging_config().keys()) + list(create_gaussian_process_config().keys())
    options += ['inference_type']
    substitutions = {'simulation:num': min(max(num_simulations, 4), 256)}
    for k, v in kwargs.items():
        if k in options:
            try: #numerical conversion from string of [bool, int, float, nD-arrays]
                substitutions[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError) as e:
                substitutions[k] = v
    feedback_on_config_overrides(substitutions, inference_prefix, domain_id)
    # build various probabilistic models
    construct_models(inference_prefix, domain_id, substitutions, desc=tag)
    # analyse performance and produce graphics
    analyse_models(inference_prefix, domain_id, substitutions, desc=tag)


if __name__=='__main__':
    """
    Program arguments:

    argv[1]  prefix of inference period (mA) <required:int>
    argv[2]  geological category (domain_id) <required:int>
    argv[3]  number of simulations <required:int>
    argv[4:] **kwargs <optional> override of default configuration parameters
             If given, the program expects space-separated arguments with one
             or more key-value pairs in the form of <key_i>=<value_i>.
             For array values, they should be enclosed in double quotes, e.g.,
             <key>="[[0.92387953,0,0.38268343], [0,1,0], [-0.38268343,0,0.92387953]]"
             kwargs["inference_type"] defaults to 'future-bench-prediction',
             however, 'in-situ-regression' is also supported.

    Example: $ python -m run_experiments 4 2301 16
    """
    inference_prefix = int(sys.argv[1])
    domain_id = int(sys.argv[2])
    num_simulations = int(sys.argv[3])
    process(inference_prefix, domain_id, num_simulations,
            **dict(arg.split('=') for arg in sys.argv[4:]))
