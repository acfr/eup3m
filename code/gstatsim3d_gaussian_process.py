# Implement Gaussian Process Regression with Sequential Gaussian Simulation
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung and Alexander Lowe <alexander.lowe@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------

import copy
import datetime
import hashlib
import multiprocessing
import numpy as np
import pandas as pd
import os
import re
import time
import warnings

from collections import defaultdict
from pathlib import Path
from pdb import set_trace as bp

from scipy.linalg import cholesky
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import QuantileTransformer

from rtc_downsample import downsample
from rtc_local_neighbourhood import find_neighbours, find_neighbours2, get_neighbourhood_mean
from rtc_trend_alignment import get_transformed_coordinates

from gstatsim3d_utils import timeit, domain_id_to_column_name

#======================================================================

class GP:
    """
    Implement Gaussian Process Regression, including the corresponding
    Sequential Gaussian Simulation and Correlated Random Field approaches.
    """

    def _create_gp_kernel(domain_id, cfg):
        """
        Compose a covariance function for use in Gaussian Process Regression

        Returns
        -------
            cov_fun : sklearn.gaussian_process.kernels object
                covariance function
            max_repeats : int
                specify number of rounds of GP learning (zero disables training optimisation)
        """
        # define a Matern or squared exponential kernel with amplitude & noise estimation enabled
        kernel_name = cfg.get('gp:kernel_name', 'SquaredExponential')
        fix_params = domain_id in cfg['gp:fixed_hyperparams']

        if fix_params: # apply previously acquired hyperparameters
            params = cfg['gp:fixed_hyperparams'][domain_id]
            if cfg.get('gp:isotropic_kernel', False) and len(params) == 5:
                # set lx=ly=lz to the geometric mean of preexisting values
                params[1:-1] = np.prod(params[1:-1])**(1./3)
                if cfg.get('gp:debug_print', False):
                  print(f"Using isotropic kernel with length scales Lx=Ly=Lz={params[1]}")
            if kernel_name == 'SquaredExponential':
                cov_fun = ConstantKernel(params[0]) * RBF(params[1:-1]) + WhiteKernel(params[-1])
            elif kernel_name == 'Matern32':
                cov_fun = ConstantKernel(params[0]) * Matern(params[1:-1], nu=1.5) + WhiteKernel(params[-1])
            else:
                raise NotImplementedError(f'{kernel_name} kernel is not supported')
            max_repeats = 0
        else:
            randseed = cfg.get('gp:randseed', 1642)
            np.random.seed(randseed)
            # GP hyperparameters: [ampl^2, kx, ky, kz, noise^2]
            # initialise length scale parameters
            params0 = np.r_[1, np.random.randint(200,500,2), np.random.randint(50,150,1), 1]
            if domain_id in cfg['gp:init_hyperparams']:
               params0 = cfg['gp:init_hyperparams'][domain_id]
            # set parameter bounds (use sklearn default if not specified)
            ampl_bounds = noise_bounds = [1.0e-5, 1.0e+5]
            if cfg.get('gp:length_scale_bounds', None) is None:
                cfg['gp:length_scale_bounds'] = [25., 10000.]
            ls_bounds = cfg['gp:length_scale_bounds']
            # set initial length_scale to a float (if isotropic) or 3-vector (if anisotropic)
            if cfg.get('gp:debug_print', False):
                if cfg.get('gp:isotropic_kernel', False):
                    print(f"An isotropic {kernel_name} kernel will be used...")
            ls_init = params0[1] if cfg.get('gp:isotropic_kernel', False) else params0[1:-1]
            if kernel_name == 'SquaredExponential':
                cov_fun = (ConstantKernel(constant_value=params0[0],
                                constant_value_bounds=ampl_bounds) *
                           RBF(length_scale=ls_init,
                                length_scale_bounds=ls_bounds) +
                           WhiteKernel(noise_level=params0[-1],
                                noise_level_bounds=noise_bounds))
            elif kernel_name == 'Matern32':
                cov_fun = (ConstantKernel(constant_value=params0[0],
                                constant_value_bounds=ampl_bounds) *
                           Matern(length_scale=ls_init,
                                  length_scale_bounds=ls_bounds, nu=1.5) +
                           WhiteKernel(noise_level=params0[-1],
                                noise_level_bounds=noise_bounds))
            else:
                raise NotImplementedError(f'{kernel_name} kernel is not supported')
            max_repeats = max(0, int(cfg.get('gp:max_repeats', 1)))

        return cov_fun, max_repeats

    def _acquire_blasthole_exploration_data(df_blasthole, df_exploration, cfg):
        """
        Read assay values from separate sources (blastholes and exploration holes)
        and compute the centroid from the combined dataset. The centroid coordinates
        will serve as the "origin" if spatial transformation is subsequently applied.
        """
        X_bh, y_bh = df_blasthole.loc[:, ['X', 'Y', 'Z']].values, df_blasthole.loc[:, 'V'].values
        X_exp, y_exp = df_exploration.loc[:, ['X', 'Y', 'Z']].values, df_exploration.loc[:, 'V'].values
        centroid = np.mean(np.concatenate((X_bh, X_exp), axis=0), axis=0)
        # keep a record of the raw blasthole and exploration data used
        (Path(cfg['info:data_dir']) / "debug").mkdir(parents=True, exist_ok=True)
        fn = Path(cfg['info:data_dir']) / "debug" / f"{cfg['gp:debug_tag']}-raw_blastholes.csv"
        pd.DataFrame(np.concatenate((X_bh, y_bh[:, np.newaxis]), axis=1)).to_csv(fn, header=False, index=False)
        fn = Path(cfg['info:data_dir']) / "debug" / f"{cfg['gp:debug_tag']}-raw_exploration.csv"
        pd.DataFrame(np.concatenate((X_exp, y_exp[:, np.newaxis]), axis=1)).to_csv(fn, header=False, index=False)
        return X_bh, X_exp, y_bh, y_exp, centroid

    def _perform_neighbourhood_mean_adjustment(X_bh, X_exp, y_bh, y_exp, cfg):
        """
        Apply relevant neighbourhood search and local mean adjustment on ore grades.

        Returns
        -------
            y_bh : numpy.array, shape (nB,)
                adjusted blasthole grades where each element has the local mean substracted from it
            y_exp : numpy.array, shape (nX,)
                adjusted exploration grades where each element has the local mean substracted from it
        """
        y_mean_bh = np.zeros(len(X_bh))
        y_mean_exp = np.zeros(len(X_exp))
        if cfg.get('gp:training_local_neighbourhood_enable', False):
            knn_min, knn_max = cfg.get('gp:local_neighbourhood_ellipse_knn_min_max', (2, 9))
            scale_enable = cfg.get('gp:local_neighbourhood_scale_enable', True)
            rotation_enable = cfg.get('gp:local_neighbourhood_rotation_enable', True)
            search_technique = cfg.get('gp:neighbourhood_search_technique', "Ellipsoid+KnnMinMax")
            search_neighbours = find_neighbours2 if search_technique == "Octant" else find_neighbours
            # conduct neighbourhood search in the transformed space, incorporating [any] rotation & scaling.
            # note: `idx_near` has shape (n, knn_max) with valid values in [0, n-1], where n=nB for
            #        "blastholes", n=nX for "exploration". For columns [knn_min:] in any row, i.e. aside
            #        from the knn_min nearest neighbors, if the distance from the query location to
            #        any neighbor exceeds 1 in the normalised search space, those indices will be
            #        assigned an out-of-bounds value of n and subsequently ignored.
            Xt_exp = get_transformed_coordinates(X_exp, scale_enable, rotation_enable, origin, domain_id, cfg)
            _, idx_near, _ = search_neighbours(Xt_exp, Xt_exp, 1. if scale_enable else None, knn_min, knn_max)
            y_mean_exp = get_neighbourhood_mean(y_exp, idx_near) # maps 2D array to 1D array
            y_exp = y_exp - y_mean_exp
            Xt_bh = get_transformed_coordinates(X_bh, scale_enable, rotation_enable, origin, domain_id, cfg)
            _, idx_near, _ = search_neighbours(Xt_bh, Xt_bh, 1. if scale_enable else None, knn_min, knn_max)
            y_mean_bh = get_neighbourhood_mean(y_bh, idx_near)
            y_bh = y_bh - y_mean_bh
            # keep a record of the neighbourhood-mean adjusted data
            fn = Path(cfg['info:data_dir']) / "debug" / f"{cfg['gp:debug_tag']}-meanadj_blastholes.csv"
            pd.DataFrame(np.concatenate((Xt_bh, y_bh[:, np.newaxis]), axis=1)).to_csv(fn, header=False, index=False)
            fn = Path(cfg['info:data_dir']) / "debug" / f"{cfg['gp:debug_tag']}-meanadj_exploration.csv"
            pd.DataFrame(np.concatenate((Xt_exp, y_exp[:, np.newaxis]), axis=1)).to_csv(fn, header=False, index=False)

        return y_bh, y_exp

    def _limit_training_samples(X_bh, X_exp, y_bh, y_exp, cfg):
        """
        Select a subset of samples for training by minimising the KL divergence
        """
        npts_before = {'blasthole': len(y_bh), 'exploration': len(y_exp)}
        if cfg.get('data:downsampling_mask', None) is None:
            cfg['data:downsampling_mask'] = dict()
            retain_bh = downsample(X_bh, y_bh, cfg.get('gp:training_max_bh_points', 0))
            retain_exp = downsample(X_exp, y_exp, cfg.get('gp:training_max_exp_points', 1024))
            mask_bh = np.zeros(len(y_bh), dtype=bool)
            mask_exp = np.zeros(len(y_exp), dtype=bool)
            mask_bh[retain_bh] = 1
            mask_exp[retain_exp] = 1
            cfg['data:downsampling_mask']['blasthole'] = mask_bh
            cfg['data:downsampling_mask']['exploration'] = mask_exp
        else:
            retain_bh = cfg['data:downsampling_mask']['blasthole'].astype(bool)
            retain_exp = cfg['data:downsampling_mask']['exploration'].astype(bool)
        X_bh, y_bh = X_bh[retain_bh, :], y_bh[retain_bh]
        X_exp, y_exp = X_exp[retain_exp, :], y_exp[retain_exp]
        npts_after = {'blasthole': len(y_bh), 'exploration': len(y_exp)}
        for k in ['blasthole', 'exploration']:
            if npts_before[k] > npts_after[k]:
                print(f"Downsampled {k} training data from {npts_before[k]} to {npts_after[k]}")
        return X_bh, X_exp, y_bh, y_exp

    def _combine_training_data(X_bh, X_exp, y_bh, y_exp, domain_id, origin, cfg):
        """
        Combine data from blasthole and exploration sources and report the training mean
        """
        X = np.concatenate((X_exp, X_bh), axis=0)
        X = get_transformed_coordinates(X, scale_enable=False,
            rotation_enable=cfg.get('gp:learning_inference_in_rotated_space', False),
            origin=origin, domain_id=domain_id, cfg=cfg)
        y = np.concatenate((y_exp, y_bh))
        # `h` represents sample height, `is_bh` indicates if measurement comes from blasthole
        #  note: these are only used for volumetric GP estimation (not for our experiments).
        h = np.concatenate((len(X_exp)*[cfg.get('gp:exp_height', 0.)],
                            len(X_bh)*[cfg.get('gp:blasthole_height', 50.)]))
        is_bh = np.concatenate((len(X_exp)*[False], len(X_bh)*[True]))
        # impose a cap on the total number of samples if desired
        max_all_points = cfg.get('gp:training_max_all_points', 1024)
        idx_keep = downsample(X, y, max_all_points)
        X, y, is_bh, h = X[idx_keep, :], y[idx_keep], is_bh[idx_keep], h[idx_keep]
        y_mean = np.mean(y)
        # keep a record of the resultant training data
        fn = Path(cfg['info:data_dir']) / "debug" / f"{cfg['gp:debug_tag']}-Xy.csv"
        pd.DataFrame(np.concatenate((X, y[:, np.newaxis], h[:, np.newaxis], is_bh[:, np.newaxis]), axis=1)).to_csv(
            fn, header=False, index=False)
        cfg['gp:training_num_blastholes'] = sum(is_bh)
        cfg['gp:training_num_exploration'] = len(is_bh) - sum(is_bh)
        return X, y, is_bh, h, y_mean

    def _gaussian_process_regressor(X, y, y_mean, cov_fun, max_repeats, domain_id, origin, cfg):
        """
        Construct a GP regressor, fitting the data means finding the nlml-optimal hyperparameters.

        Parameters
        ----------
            X : numpy.ndarray, shape=(n,3)
                x,y,z coordinates of n training samples
            y : numpy.ndarray, shape=(n,)
                the grade (assay values) of n training samples
            y_mean : float
                global mean computed from the training samples
            cov_fun : sklearn.gaussian_process.kernels
                an anisotropic covariance function obtained by _create_gp_kernel
            max_repeats: int
                number of GP kernel optimisation training rounds
            domain_id: int
                geological domain identifier
            origin : numpy.ndarray, shape(3,)
                the origin given to _combine_training_data, before any coordinates transformation
            cfg : dict
                configuration parameters
        
        Returns
        -------
            gpr : sklearn.gaussian_process.GaussianProcessRegressor object
                .ymean allows y_mean to be added back to form predictions (only needed for global neighbourhood predictions)
                .origin ensures the rotated coordinate frames are consistent (only needed for global GP inference)
            measures: dict
                'kernel_params' essentially [amplitude^2, kx, ky, kz, sigma_noise^2]
                'nlml' the minimum negative log marginal likelihood value
                'R2' coefficient of determination, viz. R2 = 1 - residual_sum_of_squares / total_sum_of_squares
                     where total_sum_of_squares = np.sum((y - y_bar)**2),
                           residual_sum_of_squares = np.sum((y - y_predict)**2)
                'attempts' number of optimisation runs
                'bh_point_count' number of blasthole samples
                'exp_point_count' number of exploration samples
                'training_point_count' total number of training samples
                'y_variance' posterior variance estimates
                'run_time' elapsed time for GP training
        """
        t0 = time.time()
        # Note: There is a .normalize_y option in the GaussianProcessRegressor class.
        #       We do not use this here to perform mean adjustment because we generally
        #       generate inferences by conditioning on a local neighbourhood. The two
        #       conceptual approaches are incongruous and it would be ludicrous to have
        #       to rescale y differently for each local neighbourhood to reconcile them.
        if cfg.get('gp:volumetric', False):
            raise NotImplementedError('This script does not support volumetric GP')
        else:
            init_state = cfg.get('gp:init_state', None)
            optimisation = "fmin_l_bfgs_b"
            if domain_id in cfg.get('gp:fixed_hyperparams', {}):
                optimisation = None # kernel parameters are kept fixed
                elapsed_time = 0.0
                print('Bypassing kernel optimisation...')
            gpr = GaussianProcessRegressor(kernel=cov_fun,
                                           optimizer=optimisation,
                                           n_restarts_optimizer=max_repeats,
                                           random_state=init_state
                                           ).fit(X, y-y_mean)
        gpr.y_mean = y_mean
        gpr.origin = origin
        elapsed_time = time.time() - t0
        # Note: according to sklearn.gaussian_process.kernels.Sum, .theta
        #       returns the flattened, log-transformed non-fixed hyperparams
        #       in the kernel composition order (see 'cov_fun')
        measures = { 'attempts': max_repeats + 1,
                     'run_time': elapsed_time,
                     'kernel_params': np.exp(gpr.kernel_.theta),
                     'R2': gpr.score(X, y),
                     'nlml': -gpr.log_marginal_likelihood(gpr.kernel_.theta),
                     'bh_point_count': cfg['gp:training_num_blastholes'],
                     'exp_point_count': cfg['gp:training_num_exploration'],
                     'training_point_count': len(X),
                     'y_variance': np.var(y)
                   }
        # If R^2 is negative, then we have failed to take into account the
        # prediction is for the residual and y_mean needs to be added back in.
        if measures['R2'] < 0:
            y_predict = gpr.predict(X)
            total_sum_of_squares = np.sum((y - np.mean(y))**2)
            residual_sum_of_squares = np.sum((y - (y_predict + gpr.y_mean))**2)
            R2 = 1.0 - residual_sum_of_squares / total_sum_of_squares
            measures['R2'] = R2

        fix_params = domain_id in cfg['gp:fixed_hyperparams']
        suppress_logging = True if fix_params else False
        GP._log_statistics(measures, cfg, suppress_logging)

        return gpr, measures

    def _log_statistics(measures, cfg, skip_logging=False):
        desc = cfg['gp:desc']
        mA = int(cfg.get('info:period_id', 00))
        inference_period = '%02d_%02d_%02d' % (mA, mA+1, mA+2)
        log_file = cfg.get('gp:log_file', None)
        n = len(measures['kernel_params'])
        kernel_params = '[{}]'.format(
                        ','.join(['%.9f' % x for x in measures['kernel_params']]))
        feedback = '{}\n' \
                   '  Blast hole count: {}\n' \
                   '  Exploration sample count: {}\n' \
                   '  Training sample count: {}\n' \
                   '  Training sample variance: {}\n' \
                   '  GP optimisation: {} attempts took {}s\n' \
                   '  optimal hyperparameters: {}\n' \
                   '  negative-log-marginal-likelihood: {}\n' \
                   '  coef. of determination: R^2={}\n'.format(
                   desc, measures['bh_point_count'], measures['exp_point_count'], measures['training_point_count'],
                   measures['y_variance'], measures['attempts'], measures['run_time'],
                   kernel_params, measures['nlml'], measures['R2'])
        print(feedback)
        if not skip_logging and log_file:
            file_path = Path(cfg['info:data_dir']) / inference_period / cfg.get('gp:log_file', "statistics.log")
            with open(file_path, 'a+') as f:
                f.write(feedback)

    def _get_data(df_blasthole, df_exploration, df_infer, cfg):
        """
        Retrieve training (blasthole [and exploration]) and query data from pandas DataFrames
        """
        X_bh, y_bh = df_blasthole.loc[:,['X','Y','Z']].values, df_blasthole.loc[:,'V'].values
        X_exp, y_exp = df_exploration.loc[:,['X','Y','Z']].values, df_exploration.loc[:,'V'].values
        X_star = df_infer.loc[:,['X','Y','Z']].values
        # Create containers for combined data (coords, values, hole_depth, blasthole_indicator)
        X = np.empty((0,3), dtype=float)
        y = np.empty((0,), dtype=float)
        # hole_depth (h), blasthole_indicator (is_bh) and query interval dimensions (dxyz)
        # serve purely as placeholders and are not currently used in our experiments
        h = np.empty((0,), dtype=float)
        is_bh = np.empty((0,), dtype=bool)
        dxyz = cfg.get('gp:query_dimensions', [0.,0.,50])
        perform_volumetric_gp = cfg.get('gp:volumetric', False)
        if 'exploration' in cfg['gp:inference_datasets']:
            X = np.concatenate((X, X_exp), axis=0)
            y = np.concatenate((y, y_exp))
            h = np.concatenate((h, len(X_exp)*[cfg.get('gp:exp_height', 0.)]))
            is_bh = np.concatenate((is_bh, len(X_exp) * [False]))
        if 'blastholes' in cfg['gp:inference_datasets']:
            X = np.concatenate((X, X_bh), axis=0)
            y = np.concatenate((y, y_bh))
            h = np.concatenate((h, len(X_bh)*[cfg.get('gp:blasthole_height', 50.)]))
            is_bh = np.concatenate((is_bh, len(X_bh) * [True]))
        if perform_volumetric_gp:
            return X, X_star, y, h, is_bh, dxyz
        else:
            return X, X_star, y
        

    @timeit
    def learning(df_blasthole, df_exploration, domain_id, debug_tag, cfg):
        """
        Use scikit-learn's GaussianProcessRegressor to train and infer block values
        """
        # Compose covariance function
        cfg['gp:debug_tag'] = debug_tag
        cov_fun, max_repeats = GP._create_gp_kernel(domain_id, cfg)

        # Load blasthole and exploration data
        X_bh, X_exp, y_bh, y_exp, origin = GP._acquire_blasthole_exploration_data(df_blasthole, df_exploration, cfg)

        # Subtract the mean from samples if the "local neighbourhood" option is enabled for training
        # - At first glance, this should be disabled as we have no way to account for non-uniform
        #   (training sample location dependent) mean shifts, and adding these subtracted mean
        #   components to the GP residual predictions unless y_mean is a constant. This would be
        #   problematic if the GP regressor obtained at the end of this learning procedure is used
        #   for local prediction. However, in this scenario, we are only interested in the optimised
        #   kernel parameters obtained here, which are used to instantiate the covariance function.
        #   Subsequently, a separate GP regressor, with K(X{i},X{i}) atuned to the zero-mean
        #   neighbourhood samples, is created for each query point X*_i to predict y_i. The local
        #   training samples (X{i}, y{i}) are selected based on the same ellipsoid criteria in the
        #   rotated and scaled search space; these will be different for each query location X*_i.
        y_bh, y_exp = GP._perform_neighbourhood_mean_adjustment(X_bh, X_exp, y_bh, y_exp, cfg)

        # Use KL divergence to subsample training set to improve computation efficiency
        X_bh, X_exp, y_bh, y_exp = GP._limit_training_samples(X_bh, X_exp, y_bh, y_exp, cfg)

        # Combine samples to form one training dataset (this may include rotation of the data)
        X, y, is_bh, h, y_mean = GP._combine_training_data(X_bh, X_exp, y_bh, y_exp, domain_id, origin, cfg)

        # Construct GP regressor
        gpr, measures = GP._gaussian_process_regressor(X, y, y_mean, cov_fun, max_repeats, domain_id, origin, cfg)

        return gpr, measures

    @timeit
    def inference_global_mean(gpr, df_infer, domain_id, cfg):
        """
        Compute posterior mean, stdev and covariance estimates using global Gaussian Process.

        Note:
          1.  The GP regressor is modelled on a zero-mean process, where the mean is
              assumed stationary and computed from all the samples seen during training.
          2.  With global GP, prediction is not applied to a zero-mean local neighbourhood
              centered at the queried points, where neighbours search might be conducted
              in a geometrically transformed (rotated and scaled) "search space".
          3.  Regardless, GP training and prediction are performed consistently in the so-
              called "inference space", where rotation may be applied if it is deemed
              necessary to align the data with geological trends.
          4.  One reason for maintaining this distinction and excluding scaling operations
              from the inference space is that the GP covariance functions will generally
              have heterogeneous length scales for the x, y and z axes, whereas scaling
              is required to account for potential scale differences to perform an elliptical
              search when implemented using a ball-based kD-tree nearest neighbours query.
          5.  Global inference is not generally used per se due to oversmoothing concerns.

        Parameters
        ----------
            gpr : sklearn.gaussian_process.GaussianProcessRegressor
                GP regressor obtained via the `learning` process
            df_infer : pandas DataFrame
                corresponds to predict locations (e.g. blocks from a domain in the bench below)
                with column names ['X','Y','Z'] for coordinates, and unknown values 'V'
            domain_id: int
                geological domain identifier
            cfg: dict
                contains configuration parameters (mainly interested in `gp:learning_inference_in_rotated_space`)

        Returns
        -------
            mean_est : numpy.ndarray, shape=(n,)
                posterior mean
            stdev_est OR cov_est : numpy.ndarray, shape=(n,) OR (n,n)
                posterior standard deviation OR covariance estimate
        """
        no_scaling = False
        compute_covariance = cfg.get('gp:compute_covariance', False)
        X_star = df_infer.loc[:, ['X', 'Y', 'Z']].values
        X_star = get_transformed_coordinates(X_star, no_scaling,
                 cfg.get('gp:learning_inference_in_rotated_space', False), gpr.origin, domain_id, cfg)

        if not compute_covariance:
            mean_est, stdev_est = gpr.predict(X_star, return_std=True)
        else:
            mean_est, cov_est = gpr.predict(X_star, return_cov=True)

        return mean_est + gpr.y_mean, stdev_est if not compute_covariance else cov_est

    @timeit
    def inference_local_mean(params, df_blasthole, df_exploration, df_infer, domain_id, cfg):
        """
        Compute posterior mean and stdev estimates using local Gaussian Process.
        Includes data wrangling (manipulating raw data into useable form)

        Note:
          1.  In contrast with the `inference_global_mean` method, the GP regressor is
              modelled on samples in a zero-mean local neighbourhood at query time.
          2.  With local GP, grade prediction is obtained using "mu_star = K(X*,X)
              inv(K(X,X) + nu^2*I) y' + mean(y)" per usual, but K(X,X) is computed using a
              subset of training samples, X, in the neighbourhood of the queried point X*,
              with y' = y - mean(y) and mean(y) computed from the chosen samples.
          3.  Each element of K(X,X), a matrix with nominal shape (knn_max,knn_max), is
              computed using a stationary covariance function that depends only on lag
              distances, using fixed `params` (previously optimised hyperparameters).
              So, GP learning is not repeated over the training data. In particular,
              gpr.fit(X, y-y_mean) is called only to compute K(X,X) using specifically the
              neighbouring samples in preparation for the next gpr.predict(X*) call.

        Parameters
        ----------
            params : list or numpy.ndarray
                optimal GP hyperparameters [ampl^2, kx, ky, kz, noise^2]
            df_blasthole : pandas DataFrame
                blasthole samples with column names ['X','Y','Z'] for coordinates, and
                known grade values 'V' already subject to normal score transformation
            df_exploration : pandas DataFrame
                largely a placeholder for exploration hole samples with the same columns
            df_infer : pandas DataFrame
                corresponds to predict locations (e.g. blocks from a domain in the bench below)
                with column names ['X','Y','Z'] for coordinates, and unknown values.
            domain_id: int
                geological domain identifier
            cfg: dict
                contains configuration parameters

        Returns
        -------
            mean_est : numpy.ndarray, shape=(n,)
                posterior mean
            stdev_est : numpy.ndarray, shape=(n,)
                posterior standard deivation
        """
        X, X_star, y = GP._get_data(df_blasthole, df_exploration, df_infer, cfg)

        # Instantiate covariance function with learned hyperparameters
        cfg['gp:fixed_hyperparams'][domain_id] = params
        cov_fun = GP._create_gp_kernel(domain_id, cfg)[0]

        # Nominate neighbourhood search fuction
        search_technique = cfg.get('gp:neighbourhood_search_technique', "Ellipsoid+KnnMinMax")
        search_neighbours_fn = find_neighbours2 if search_technique == "Octant" else find_neighbours

        # Specify known samples and predict locations in the neighbourhood search space
        # and GP inference space (see notes 2-4 in `inference_global_mean` doc string)
        # Note: `origin` does not have to match the one used during GP learning because
        #      a) we are forming new local training set for each prediction
        #      b) stationary kernels care only about distances between samples
        origin = np.mean(X, axis=0)
        search_scale_enable = cfg.get('gp:local_neighborhood_scale_enable', True)
        search_rotation_enable = cfg.get('gp:local_neighborhood_rotation_enable', True)
        gp_rotation_enable = cfg.get('gp:learning_inference_in_rotated_space', False)
        # Training and query locations in local neighbourhood search space
        # _search suffix denotes local neighborhood search space, which may be rotated and
        # scaled depending on ellipsoid definition for the domain
        X_search = get_transformed_coordinates(X, search_scale_enable,
                   search_rotation_enable, origin, domain_id, cfg)
        X_star_search = get_transformed_coordinates(X_star, search_scale_enable,
                        search_rotation_enable, origin, domain_id, cfg)
        # Training and query locations in GP inference space
        # _gp suffix denotes the GP inference space, which may be rotated depending
        # on the cfg and ellipsoid specification for the given domain
        X_gp = get_transformed_coordinates(X, False,
               gp_rotation_enable, origin, domain_id, cfg)
        X_star_gp = get_transformed_coordinates(X_star, False,
                    gp_rotation_enable, origin, domain_id, cfg)
        del(X)

        # Loop over query locations
        mean_predicted = np.full(len(X_star), np.nan)
        sd_predicted = np.full(len(X_star), np.nan)
        knn_min, knn_max = cfg.get('gp:local_neighborhood_ellipse_knn_min_max', (2,9))
        knn_params = {'knn_search_scale': 1. if search_scale_enable else None,
                      'knn_min': knn_min, 'knn_max': knn_max, 'covariance_obj': cov_fun}
        tree = None
        
        for i, (X_star_search_i, X_star_gp_i) in enumerate(zip(X_star_search, X_star_gp)):
            m, s, tree = GP._infer_single_point(X_search, X_gp, y, X_star_search_i,
                             X_star_gp_i, tree, search_neighbours_fn, knn_params)
            mean_predicted[i] = m
            sd_predicted[i] = s

        return mean_predicted, sd_predicted

    def _infer_single_point(X_search, X_gp, y, X_star_search_i, X_star_gp_i, tree, knn_search_fn, parms):
        """
        Macro-like function for performing a single localised GP query

        Parameters
        ----------
            X_search : numpy.ndarray, shape=(nT,3)
                x,y,z coordinates of all training samples in (rotated+scaled) search space
            X_gp : numpy.ndarray, shape=(nT,3)
                x,y,z coordinates of all training samples in (rotated) inference space
            y : numpy.ndarray, shape=(nT,)
                known values for the predicted variable from all training samples
            X_star_search_i : numpy.ndarray, shape=(3,)
                x,y,z coordinates for the queried point given in the search space
            X_star_gp_i : numpy.ndarray, shape=(3,)
                x,y,z coordinates for the queried point given in the inference space
            tree : None or scipy.spatial cKDTree
                used to make nearest neighbour query
            knn_search_fn : method
                neighbourhood search function, default: `find_neighbourhood`
                    `find_neighbourhood` uses elliptical search with (knn_min, knn_max) sample constraint
                    `find_neighbourhood2` is similar, but prefers samples taken from different octants
            parms : dict
                specifies the `cov_fun` and `knn_seaerch_scale`, `knn_min`, `knn_max`

        Returns
        -------
            m : float
                mean value predicted by local GP regression
            s : float
                a standard deviation estimate at the queried location
            tree : scipy.spatial cKDTree
                kD-tree for re-use
        """
        # Find training data in local neighbourhood
        _, idx_near, tree = knn_search_fn(X_search, [X_star_search_i],
                            parms['knn_search_scale'], parms['knn_min'],
                            parms['knn_max'], tree)
        # Retain valid samples within scope
        idx_near = idx_near[idx_near < len(y)]
        X_gp_i, y_i = X_gp[idx_near,:], y[idx_near]
        y_mean = np.mean(y_i)
        # Perform point-wise prediction (note: volumetric GP is not supported)
        # - setting `optimizer` to None fixes the hyperparameters (disables optimisation)
        # - internally, we expect the covariance matrix K(X,X) to be recomputed though
        #   given the local training data (X=X_gp_i, y=y_i) before making a prediction
        gpr = GaussianProcessRegressor(kernel=parms['covariance_obj'],
              optimizer=None).fit(X_gp_i, y_i - y_mean)
        m, s = gpr.predict([X_star_gp_i], return_std=True)
        m += y_mean
        return m[0], s[0], tree

    @timeit
    def inference_sgs(params, df_blasthole, df_exploration, df_infer, domain_id, cfg, randseed=None, desc=""):
        """
        Perform Sequential Gaussian Simulation using Gaussian Process Regression

        Pre-condition: Column 'V' is assumed to contain Gaussian distributed (normal score transformed) values.
        Parameters and returned results are similar to those in `inference_local_mean`
        """
        X, X_star, y = GP._get_data(df_blasthole, df_exploration, df_infer, cfg)

        # Instantiate covariance function with learned hyperparameters
        cfg['gp:fixed_hyperparams'][domain_id] = params
        cov_fun = GP._create_gp_kernel(domain_id, cfg)[0]

        # Nominate neighbourhood search fuction
        search_technique = cfg.get('gp:neighbourhood_search_technique', "Ellipsoid+KnnMinMax")
        search_neighbours_fn = find_neighbours2 if "Octant" in search_technique else find_neighbours

        # Specify known samples and predict locations in search space and inference space
        origin = np.mean(X, axis=0)
        search_scale_enable = cfg.get('gp:local_neighborhood_scale_enable', True)
        search_rotation_enable = cfg.get('gp:local_neighborhood_rotation_enable', True)
        gp_rotation_enable = cfg.get('gp:learning_inference_in_rotated_space', False)
        X_search = get_transformed_coordinates(X, search_scale_enable,
                   search_rotation_enable, origin, domain_id, cfg)
        X_star_search = get_transformed_coordinates(X_star, search_scale_enable,
                        search_rotation_enable, origin, domain_id, cfg)
        X_gp = get_transformed_coordinates(X, False,
               gp_rotation_enable, origin, domain_id, cfg)
        X_star_gp = get_transformed_coordinates(X_star, False,
                    gp_rotation_enable, origin, domain_id, cfg)
        del(X)

        # Create randomised path, set the seed for reproducible results
        if randseed is not None:
            np.random.seed(randseed)
        path_indices = np.arange(len(X_star))
        np.random.shuffle(path_indices)

        # Perform conditional evaluation following a random path
        y_sim = np.full(len(X_star), np.nan)
        knn_min, knn_max = cfg.get('gp:local_neighborhood_ellipse_knn_min_max', (2,9))
        knn_params = {'knn_search_scale': 1. if search_scale_enable else None,
                      'knn_min': knn_min, 'knn_max': knn_max, 'covariance_obj': cov_fun}
        tree = None
        for i in path_indices:
            mu, sigma, tree = GP._infer_single_point(X_search, X_gp, y, X_star_search[i],
                              X_star_gp[i], tree, search_neighbours_fn, knn_params)
            y_sim[i] = np.random.normal(mu, sigma, 1)
            # Add processed node/value to conditional data
            X_search = np.concatenate((X_search, [X_star_search[i]]), axis=0)
            X_gp = np.concatenate((X_gp, [X_star_gp[i]]), axis=0)
            y = np.concatenate((y, [y_sim[i]]), axis=0)

        return y_sim

    @timeit
    def inference_cfr(gpr, df_infer, domain_id, cfg, gp_mean=None, gp_L=None, randseed=None, desc=""):
        """
        Generate Conditional Random Field using Gaussian Processes
        """
        if gp_L is None:
            # Perform Cholesky decomposition
            cfg['gp:compute_covariance'] = True
            gp_mean, gp_cov = GP.inference_global_mean(gpr, df_infer, domain_id, cfg)
            gp_L = cholesky(gp_cov, lower=True)

        # Multiply L (lower triangle matrix) by w (a vector of uncorrelated
        # random numbers drawn from a unit normal distribution)
        if randseed is not None:
            rng = np.random.default_rng(seed=randseed)

        w = rng.normal(loc=0.0, scale=1.0, size=gp_L.shape[0])
        y_sim = np.matmul(gp_L, w) + gp_mean
        return y_sim, gp_mean, gp_L


#======================================================================
class GPManager:
    """
    Provide an interface for carrying out sequential simulations based on GP regression
    and computing multiple spatially correlated random fields using GP cholesky decomposition.

    Note: Apart from invoking the SGS or CRF routine in a FOR loop, it also performs
    data wrangling (converting raw data into useful forms, e.g. applying normal score
    transformation before GP learning and after various GP predictions), selection of
    model-period and domain dependent random seeds (to produce deterministic random paths).
    """

    def gaussian_process_simulations(method, df_blasthole, df_exploration, df_infer, cfg):
        """
        API to perform standard GP regression, multiple sequential simulations using GPR,
        or multiple conditional random field simulations using GP cholesky decomposition.

        Parameters
        ----------
            method : str
                specifies the synthesis method, one of "GPR(L)", "GPR(G)", "GP-SGS" or "GP-CRF"
            df_blasthole : pandas DataFrame
                corresponds to blasthole training data, with column names ['X','Y','Z']
                for coordinates, 'V' (or 'PL_CU') for values in the specified domain
            df_exploration : pandas DataFrame
                corresponds to exploration training data, with column names ['X','Y','Z']
                for coordinates, 'V' (or 'PL_CU') for values in the specified domain
            df_infer : pandas DataFrame
                corresponds to predict locations (e.g. blocks from a domain in the bench below)
                with column names ['X','Y','Z'] for coordinates, and unknown values 'V'
            cfg: python dict
                specifies the parameters and chosen options for a given experiment
                'gp:inference_datasets' : list(str)
                    valid strings are 'blastholes' and or 'exploration'
                'gp:volumetric': bool
                    please set to False, as volumetric GP is not supported by this script
                'gp:exp_height': float
                    length of exploration hole measurement (for volumetric GP), default: 0.
                'gp:blasthole_height' : float
                    length of blasthole measurement (for volumetric GP), default: 50.
                'gp:training_max_bh_points' : int
                    max number of blastholes to use during training, default: 0
                'gp:training_max_exp_points' : int
                    max number of exploration holes to use during training, default: 0
                'gp:training_max_all_points' : int
                    max number of blast/exploration hole samples combined, default: 1000000
                'gp:training_region_margin': list(float)
                    a margin around the query region (block centroids), outside of which to
                    clip training data, default: [3000., 3000., 3000.]
                'gp:training_local_neighborhood_enable' : bool
                    search for samples in the local neighbourhood to use for local mean
                    adjustment during training, default: False
                'gp:inference_local_neighborhood_enable' : bool
                    search for samples in the local neighbourhood to use for inference
                'gp:local_neighborhood_ellipse_knn_min_max': tuple(int)
                    min/max samples in a local neighbourhood, default: (2,9)
                'gp:local_neighborhood_scale_enable' : bool
                    perform local neighbourhood search using an ellipse, default: True
                'gp:local_neighborhood_rotation_enable' : bool
                    perform local neighbourhood search in rotated space, default: True
                'gp:learning_inference_in_rotated_space' : bool
                    carry out gp training/inference in rotated space
                'gp:ellipse_definitions_filename' : str
                    file specifying ellipse orientation and search scale for various domains
                'gp:ellipse_default_radius' : tuple(float)
                    ellipse radii for undefined domain, default: (300., 300., 300.)
                'gp:ellipse_default_rotation' : tuple(float)
                    orientation for undefined domain, default: (0., 0., 0.)
                'gp:neighbourhood_search_technique': str
                    "Ellipsoid+KnnMinMax" - this default option conducts elliptical
                    neighbourhood search, including knn_min samples unconditionally and
                    return up to knn_max samples within the search range if possible
                    "Ellipsoid+Octant" - as above, but with priority given to selecting
                    (knn_max // 8) samples from each individual octant (assuming coords
                    are centered at the query point), then choosing the remaining samples
                    from the nearest neighbours yet to be picked if there is a short fall.
                'gp:kernel_name' : str
                    name of GP kernel (covariance function), one of ['SquaredExponential','Matern32']
                'gp:isotropic_kernel' : bool
                    whether to set all length scales to the same value, default: False (anisotropic kernel)
                'gp:init_hyperparams' : dict
                    initial values [ampl_0^2, kx_0, ky_0, kz_0, nu_0^2] to use during learning
                'gp:fixed_hyperparams' : dict
                    preset kernel hyperparameters, mapping domain_id to [ampl^2,k1,k2,k3,nu^2]
                    when the key is present, kernel hyperparameters will be fixed rather than re-learned
                'gp:max_repeats' : int
                    number of optimisation rounds during GP learning, default: 3
                'gp:init_state': int
                    for reproducible results, default: 2633
                'gp:clear_previous_log': bool
                    overwrite existing logs, default: True
                'gp:log_file' : str
                    where GP learning statistics are written to
                'domain_column_name' : str
                    name for the domain column name in blastholes.csv, default: 'lookup_domain'
                'simulation:filename_template' : str
                    name for file where simulated values will be written, default: 'gstatsim3d_gp.csv'
                'simulation:num' : int
                    total number of sequential simulations
                'info:period_id' : int
                    time period identifier, the mA part in "mA_mB_mC" where for instance
                    mA = 4 implies we are inferring values using causal data for
                    the operation period (weeks) denoted "4_5_6"
                'info:domain_id' : int
                    four digit domain id "<limbzone><gradezone><porphyryzone><rocktype>"
                'info:block_columns_to_copy' : str
                    columns to copy from blocks_to_estimate.csv into the simulation results file
                'info:data_dir' : str
                    input data directory

        Effects
        -------
        1)  df_infer (itself a data frame with indices corresponding to a specific domain)
            will have new column name "<method>_<i>?" with simulated values added to it.
            <method> is one of ['GPR(L)', 'GPR(G)', 'GP-SGS', 'GP-CRF']
            <i> denotes iteration i in the sequential simulation (not applicable to 'GPR(*)').
        2)  cfg['simulation:column_names'] will contain a list of column names for simulated values
        3)  random states and identifiers will be added to the cfg, in the form of
            ['simulation:period_domain_id'] that resembles "{inference_period}:{domain_key}",
            ['simulation:period_domain_initial_state']: a SHA256 modulo 10**8 hash value,
            ['simulation:path_seeds']: list of M random seeds chosen for each simulation.
        """
        # Label x,y,z columns consistently
        df_blasthole = df_blasthole.rename(columns = {'EAST':'X', 'NORTH':'Y', 'RL':'Z'})
        df_exploration = df_exploration.rename(columns = {'midx':'X', 'midy':'Y', 'midz':'Z'})
        region_bounds = GPManager._get_region_bounds(df_infer, cfg)
        max_num_bh = cfg.get('gp:training_max_bh_points', 0)
        df_blasthole = GPManager._region_clip(df_blasthole, *region_bounds, max_num_bh)
        df_exploration = GPManager._region_clip(df_exploration, *region_bounds)
        grade_column_bh = 'PL_CU' if 'PL_CU' in df_blasthole.columns else 'V'
        grade_column_exp = 'PL_CU' if 'PL_CU' in df_exploration.columns else 'V'
        raw_vals = np.concatenate((df_blasthole[grade_column_bh].values,
                                   df_exploration[grade_column_exp].values)).reshape(-1,1)

        # Normal score transformation - convert values to Gaussian data and store in 'V'
        # Perform this jointly on blastholes and exploration, then push into individual df.
        nst_preference = cfg.get('gp:apply_normal_score_transform', False)
        apply_normal_score_xform = True if method in ['GP-SGS', 'GP-CRF'] else nst_preference
        if len(raw_vals) < cfg.get('transformation:nst_min_sample_size', 30):
            apply_normal_score_xform = False

        if apply_normal_score_xform:
            rstate = cfg.get('transformation:normal_score_randseed', 8725)
            nst_trans = QuantileTransformer(n_quantiles=min(len(raw_vals), 1000),
                        output_distribution='normal', random_state=rstate).fit(raw_vals)
            combined_vals = nst_trans.transform(raw_vals).flatten()
        else:
            combined_vals = raw_vals.flatten().astype(np.float64)
        split = len(df_blasthole)
        df_blasthole.loc[:,'V'] = combined_vals[:split]
        df_exploration.loc[:,'V'] = combined_vals[split:]

        # Experiment description
        domain_id = int(cfg.get('info:domain_id', 8888))
        domain_key = domain_id_to_column_name(domain_id)
        mA = int(cfg.get('info:period_id', 1))
        inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_serial = GPManager._make_experiment_serial(method, cfg)
        cfg['gp:mean_col_name'] = f"mean_rtc_{experiment_serial}"
        cfg['gp:stdev_col_name'] = f"stdev_rtc_{experiment_serial}"
        cfg['gp:desc'] = f'[{timestamp}] directory: {inference_period}, domain: {domain_id}, experiment_serial={experiment_serial}'
        cfg['internal:experiment_serial'] = experiment_serial

        # generate time-period and domain specific random seeds to
        # produce unique and reproducible random paths
        cfg['simulation:period_domain_id'] = s = f"{inference_period}:{domain_key}"
        cfg['simulation:period_domain_initial_state'] = \
            int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
        rng = np.random.default_rng(seed=cfg['simulation:period_domain_initial_state'])
        num_simulations = int(cfg.get('simulation:num', 20))
        cfg['simulation:path_seeds'] = rng.integers(low=0, high=2**31, size=num_simulations)

        if cfg.get('simulation:bypass', False):
            cfg['simulation:bypass'] = False
            return

        # Learning phase
        debug_tag = str(domain_id)
        # - attempt to retrieve optimal hyperparameters for this domain if they already exist
        hyperparams = GPManager._read_hyperparameters_from_csv(cfg)
        if hyperparams is not None:
            cfg['gp:fixed_hyperparams'][domain_id] = np.array(hyperparams)
            print(f"For domain {domain_id}, using fixed hyperparams: {hyperparams}")
        else: #- erase any pre-existing values that might have been left there
            cfg['gp:fixed_hyperparams'].pop(domain_id, None)

        gpr, measures = GP.learning(df_blasthole, df_exploration, domain_id, debug_tag, cfg)
        kernel_params = measures['kernel_params']
        cfg['timing:learn'] = measures['run_time']

        if hyperparams is None:
            learning_space = GPManager._learning_space_designation(cfg)
            kernel_name = cfg.get('gp:kernel_name', 'SquaredExponential')
            hyperparams = defaultdict(list)
            hyperparams['timestamp'].append(timestamp)
            hyperparams['learning_space'].append(learning_space)
            hyperparams['kernel_name'].append(kernel_name)
            hyperparams['inference_period'].append(inference_period)
            hyperparams['domain'].append(domain_id)
            hyperparams['experiment_serial'].append(experiment_serial)
            hyperparams['kernel_params'].append(kernel_params)
            if cfg.get('gp:log_nlml_r2_with_hyperparams', False):
                hyperparams['nlml'].append(measures['nlml'])
                hyperparams['R2'].append(measures['R2'])

            GPManager._write_hyperparameters_to_csv(hyperparams, cfg)

        # Prediction phase
        if 'GPR' in method:
            t0 = time.time()
            # perform GP regression using local or global mean approach
            if method == 'GPR(G)':
                local_cfg = copy.deepcopy(cfg)
                local_cfg['gp:compute_covariance'] = False
                mean_est, stdev_est = GP.inference_global_mean(gpr, df_infer, domain_id, local_cfg)
            else:
                mean_est, stdev_est = GP.inference_local_mean(kernel_params, df_blasthole,
                                      df_exploration, df_infer, domain_id, cfg)
            if apply_normal_score_xform:
                mean_est = nst_trans.inverse_transform(mean_est.reshape(-1,1))
                if cfg.get('gp:nst_stdev_correction_enable', True):
                    # Include first-order correction for the stdev estimate
                    # V[f(X)] ~= f'(mu)[f'(mu) - mu*f"(mu)]*sigma^2
                    #         - [f"(mu)^2*mu + (f'(mu)-mu*f"(mu))*f"(mu)]*skewness*sigma^3
                    #         + (1/4)*f"(mu)^2*(kurtosis-1)*sigma^4
                    #         ~= f'(mu)^2 * V[X]
                    eps = cfg.get('gp:nst_stdev_correction_epsilon', 1e-9)
                    raw_vals_x = raw_vals.flatten()
                    transformed_vals_fx = combined_vals.flatten()
                    iSort = np.argsort(raw_vals_x)
                    gradients = np.diff(transformed_vals_fx[iSort]) / (eps + np.diff(raw_vals_x[iSort]))
                    gradients_trunc = np.sort(gradients[gradients > 0])
                    n = len(gradients_trunc)
                    m = 0.5 * (n - 1)
                    f_prime_estim = np.median(gradients_trunc)
                    stdev_est *= (1.0 / f_prime_estim)
                    if cfg.get('gp:nst_stdev_correction_debug', False):
                        print(f"gradient: non-positive elements={sum(gradients <= 0)}/{len(gradients)}")
                        print(f"gradient: percentile={np.percentile(gradients_trunc, np.linspace(0,100,21))}")
                        print(f"gradient: median={f_prime_estim}, index:{m}/{n}")
                        print(f"gradient: window {gradients_trunc[int(m)-4:int(m)+4]}")
                else:
                    warnings.warn("GP stdev estimate has not been compensated for NST, consider "
                                  "setting `gp:apply_normal_score_transform` to False")
            cfg['timing:inference'] = time.time() - t0
            return mean_est, stdev_est
        else: # run simulations
            if method not in ['GP-SGS', 'GP-CRF']:
                raise NotImplementedError(f'Uunsupported simulation method {method}')
            cfg['simulation:column_names'] = []
            gp_mean, gp_L = None, None
            t0 = time.time()
            for i in np.arange(num_simulations):
                col_name = f"{method.replace('-','_')}_" + str(i)
                cfg['simulation:column_names'].append(col_name)
                cfg['gp:debug_print'] = True if i==0 else False
                if method == 'GP-SGS':   #Sequential Gaussian Simulation using local GP
                    y_sim = GP.inference_sgs(kernel_params, df_blasthole, df_exploration,
                            df_infer, domain_id, cfg, cfg['simulation:path_seeds'][i], desc=str(i))
                elif method == 'GP-CRF': #simulate Gaussian Conditional Random Field
                    y_sim, gp_mean, gp_L = GP.inference_cfr(gpr, df_infer, domain_id, cfg,
                                           gp_mean, gp_L, cfg['simulation:path_seeds'][i], desc=str(i))
                # reverse normal score transformation
                if apply_normal_score_xform:
                    v_star = nst_trans.inverse_transform(y_sim.reshape(-1,1))
                else:
                    v_star = y_sim
                df_infer.loc[:,col_name] = v_star

            cfg['timing:inference'] = time.time() - t0
            return df_infer[cfg['simulation:column_names']]

    def _get_region_bounds(df_k, cfg):
        margin = cfg.get('gp:training_region_margin', None)
        bl = df_k.loc[:, ['X','Y','Z']].min().values - margin
        tr = df_k.loc[:, ['X','Y','Z']].max().values + margin
        return bl, tr

    def _region_clip(df, bl, tr, min_point_count=0):
        xyz = df.loc[:, ['X','Y','Z']].values
        idx_in_bounds = np.all((xyz > bl[np.newaxis,:]) & (xyz < tr[np.newaxis,:]), axis=1)
        if np.sum(idx_in_bounds) < min_point_count:
            _, idx_near, _ = find_neighbours(xyz, (tr+bl)/2., [1.,1.,1.], min_point_count, min_point_count, p_distance_exponent=np.inf)
            idx_near = idx_near.flatten()
            idx_near = idx_near[idx_near < len(df)]
            return df.iloc[idx_near]
        else:
            return df.loc[idx_in_bounds]

    def _remove_chars(s, chars):
        for c in chars:
            s = s.replace(c, '')
        return s

    def _make_experiment_serial(method, cfg):
        # Return a descriptive string of the form <method>[r]_[g|l][be]t_[l][b]i
        # where <method> = map(_remove_chars, m.replace('GPR','GP')) and m is one of
        # {"GPR(L)", "GPR(G)" "GP-SGS", "GP-CRF"}, see `gaussian_process_simulations` docstr
        # [r] is included when cfg['gp_rotation_enable'] is True
        # [g] and [l] generally indicate global or local GP approaches
        # [b] and [e] indicate whether blasthole and/or exploration data is used
        # 't' and 'i' suffices indicate the setting for training and inference
        prefix = cfg.get('gp:rotation_serial_prefix', '')
        if len(prefix) > 0:
            prefix += '_'
        method_ = GPManager._remove_chars(method.lower().replace('gpr','gp'), ['-','(',')'])
        r = 'r' if cfg.get('gp:learning_inference_in_rotated_space', False) else ''
        train_prefix = 'l' if cfg.get('gp:training_local_neighborhood_enable', False) else 'g'
        train_blast = 'b' if cfg.get('gp:training_max_bh_points', 0) > 0 else ''
        train_exp = 'e' if cfg.get('gp:training_max_exp_points', 0) > 0 else ''
        infer_prefix = 'l' if cfg.get('gp:inference_local_neighborhood_enable', False) else 'g'
        infer_blast = 'b' if "blastholes" in cfg['gp:inference_datasets'] else ''
        infer_exp = 'e' if "exploration" in cfg['gp:inference_datasets'] else ''
        second_term = f"{train_prefix}{train_blast}{train_exp}t"
        third_term = f"{infer_prefix}{infer_blast}{infer_exp}i"
        transform = "_nst" if cfg.get('gp:apply_normal_score_transform', False) else ""
        return f"{prefix}{method_}{r}_{second_term}_{third_term}{transform}"

    def _format_kernel_params(x):
        re_cascade = lambda y, pA, pB, pC: re.sub(pC, '', re.sub(pB, '', re.sub(pA, ' ', y)))
        remove_enclosed_typename = lambda y: re.sub(r'np.float\d*\(', '', re.sub(r'\)', '', y))
        scientific_to_float = lambda y: ' '.join(['%.12g' % float(i) for i in re.split(r',\s*| ', y)])
        return '[' + scientific_to_float(remove_enclosed_typename(
                     re_cascade(str(x), r'\s+', r'\s*\[\s*', r'\s*\]\s*'))) + ']'

    def _learning_space_designation(cfg):
        return 'rotated' if cfg.get('gp:learning_inference_in_rotated_space', True) else 'not_rotated'

    def _write_hyperparameters_to_csv(hyperparams_dict, cfg):
        csvname = cfg.get('gp:hyperparams_csv_file', 'gstatsim3d_optimised_hyperparameters_gp.csv')
        fname = Path(cfg['info:data_dir']) / csvname
        df = pd.DataFrame(hyperparams_dict)
        df['kernel_params'] = df['kernel_params'].apply(GPManager._format_kernel_params)
        if os.path.isfile(fname):
            # check if entry with same attributes already exists
            empty_file = os.stat(fname).st_size == 0
            if not empty_file:
                dfr = pd.read_csv(fname, header=0, sep=',')
                mA = int(cfg.get('info:period_id', 1))
                learning_space = GPManager._learning_space_designation(cfg)
                kernel_name = cfg.get('gp:kernel_name', 'SquaredExponential')
                inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
                domain_id = int(cfg['info:domain_id'])
                experiment_serial = cfg['internal:experiment_serial']
                matches = dfr.loc[(dfr['learning_space']==learning_space) &
                                  (dfr['kernel_name']==kernel_name) &
                                  (dfr['inference_period']==inference_period) &
                                  (dfr['experiment_serial']==experiment_serial) &
                                  (dfr['domain']==domain_id)].index
            if empty_file or len(matches) == 0:
                incl_hdr = True if empty_file else False
                df.to_csv(fname, index=False, float_format='%.9g', header=incl_hdr, mode='a')
        else:
            df.to_csv(fname, index=False, float_format='%.9g', header=True, mode='w')

    def _read_hyperparameters_from_csv(cfg):
        csvname = cfg.get('gp:hyperparams_csv_file', 'gstatsim3d_optimised_hyperparameters_gp.csv')
        fname = Path(cfg['info:data_dir']) / csvname
        parms = None
        try: #retrieve kernel params for specific (learning_space, inference period, domain, experiment serial)
            df = pd.read_csv(fname, header=0, sep=',')
            mA = int(cfg.get('info:period_id', 1))
            learning_space = GPManager._learning_space_designation(cfg)
            kernel_name = cfg.get('gp:kernel_name', 'SquaredExponential')
            inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
            domain_id = int(cfg['info:domain_id'])
            experiment_serial = cfg['internal:experiment_serial']
            cfg['gp:mean_col_name'] = f"mean_rtc_{experiment_serial}"
            cfg['gp:stdev_col_name'] = f"stdev_rtc_{experiment_serial}"
            row = df.loc[(df['learning_space']==learning_space) &
                         (df['kernel_name']==kernel_name) &
                         (df['inference_period']==inference_period) &
                         (df['experiment_serial']==experiment_serial) &
                         (df['domain']==domain_id)].index
            strip = lambda x: re.sub(r'\s*\]\s*', '', re.sub(r'\s*\[\s*', '', re.sub(r'\s+', ' ', x)))
            parms = [float(x) for x in strip(df.loc[row[0],'kernel_params']).split()]
        except:
            pass
        return parms

__all__ = ['KrigingRegression', 'KrigingManager']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]
