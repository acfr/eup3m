# Implement Simple and Ordinary Kriging and corresponding Sequential Gaussian Simulation routines
#
# Note: This script extends some of the code from the GStatSim package
#       to work with irregular 3D data. https://pypi.org/project/gstatsim/ was
#       contributed by Emma MacKie and made available under the MIT License.
#       The contents here are covered by the BSD-3-Clause License.
#
##-------------------------------------------------------------------------------
## MIT License
##
## Copyright (c) 2022 Emma MacKie
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##-------------------------------------------------------------------------------
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
##-------------------------------------------------------------------------------


import datetime
import hashlib
import os
import numpy as np
import pandas as pd
import skgstat as skg
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from pdb import set_trace as bp
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

from rtc_downsample import downsample
from gstatsim3d_utils import timeit, domain_id_to_column_name, NearestNeighbor, Covariance

#======================================================================

class KrigingRegression:
    """
    Implement Simple and Ordinary Kriging and Sequential Gaussian Simulation based upon these
    """

    @timeit
    def skrige(xyz_predict, df_known, num_points, vario, radius, quiet=False):
        """
        Simple kriging regression

        Parameters
        ----------
            xyz_predict : numpy.ndarray
                x,y,z coordinates of prediction grid, or query points to be estimated
            df_known : pandas DataFrame
                with column names ['X','Y','Z'] for coordinates, 'V' for values
                data frame of conditioning data
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [nugget, range, sill, nu, vtype, R, S]
                nugget, range, sill are common to all vtypes, supported vtypes
                include {'Exponential', 'Spherical', 'Matern', 'Gaussian'}
                nu describes the smoothness of the Matern kernel (e.g. 1.5)
                R is the rotation matrix, either numpy.array of shape (3,3) or None
                S is the scaling array, either numpy.array of shape (3,) or None
            radius : int, float
                search radius (generally, this being half the effective range)
            quiet : bool
                If False, a progress bar will be printed to the console.
               Default is False

        Returns
        -------
            est_sk : numpy.ndarray
                simple kriging estimate for each coordinate in xyz_predict
            var_sk : numpy.ndarray 
                simple kriging variance 
        """
        
        # unpack variogram parameters
        nugget, effective_range, sill, nu, vtype, R, S = vario
        var_1 = sill
        mean_1 = df_known['V'].mean()
        est_sk = np.zeros(len(xyz_predict)) 
        var_sk = np.zeros(len(xyz_predict))

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(xyz_predict, position=0, leave=True))
        else:
            _iterator = enumerate(xyz_predict)

        # for each coordinate requiring prediction
        for i, xyz_star in _iterator:
            matches = np.sum(np.isclose(df_known[['X','Y','Z']].values, xyz_star), axis=1)==3
            if any(matches):
                # query location corresponds to a known sample
                est_sk[i] = df_known['V'].values[np.where(matches)[0][0]]
                var_sk[i] = 0
            else:
                # gather nearest points within radius
                nearest = NearestNeighbor.nearest_neighbor_search(
                          radius, num_points, xyz_star, df_known[['X','Y','Z','V']])
                xyz_known = nearest[:, :-1]
                vals_known = nearest[:,-1]
                # covariance between data
                covariance_matrix = Covariance.make_covariance_matrix(
                                    xyz_known, vario)
                # covariance between data and unknown
                covariance_array = Covariance.make_covariance_array(
                                    xyz_known, xyz_star, vario)
                # solve kriging equations
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond=None)

                est_sk[i] = mean_1 + (np.sum(k_weights * (vals_known[:] - mean_1)))
                var_sk[i] = var_1 - np.sum(k_weights * covariance_array)
                var_sk[var_sk < 0] = 0

        return est_sk, var_sk

    @timeit
    def okrige(xyz_predict, df_known, num_points, vario, radius, quiet=False):
        """
        Ordinary kriging regression

        Parameters
        ----------
        Refer to `skrige`

        Returns
        -------
            est_ok : numpy.ndarray
                ordinary kriging estimate for each coordinate in xyz_predict
            var_ok : numpy.ndarray
                ordinary kriging variance 
        """

        # unpack variogram parameters
        nugget, effective_range, sill, nu, vtype, R, S = vario
        var_1 = sill
        est_ok = np.zeros(len(xyz_predict)) 
        var_ok = np.zeros(len(xyz_predict))

        # build the iterator
        if not quiet:
            _iterator = enumerate(tqdm(xyz_predict, position=0, leave=True))
        else:
            _iterator = enumerate(xyz_predict)

        for i, xyz_star in _iterator:
            matches = np.sum(np.isclose(df_known[['X','Y','Z']].values, xyz_star), axis=1)==3
            if any(matches):
                # query location corresponds to a known sample
                est_ok[i] = df_known['V'].values[np.where(matches)[0][0]]
                var_ok[i] = 0
            else:
                # find nearest data points
                nearest = NearestNeighbor.nearest_neighbor_search(
                          radius, num_points, xyz_star, df_known[['X','Y','Z','V']])
                xyz_known = nearest[:, :-1]
                vals_known = nearest[:,-1]
                local_mean = np.mean(vals_known)
                n_neighbors = len(nearest)
                # covariance between data
                covariance_matrix = np.zeros((n_neighbors+1, n_neighbors+1)) 
                covariance_matrix[:n_neighbors,:n_neighbors] = \
                    Covariance.make_covariance_matrix(xyz_known, vario)
                covariance_matrix[-1,:n_neighbors] = 1
                covariance_matrix[:n_neighbors,-1] = 1
                # covariance between data and unknown
                covariance_array = np.zeros(n_neighbors+1)
                k_weights = np.zeros(shape=(n_neighbors+1))
                covariance_array[:n_neighbors] = \
                    Covariance.make_covariance_array(xyz_known, xyz_star, vario)
                covariance_array[n_neighbors] = 1 
                covariance_matrix.reshape(((n_neighbors+1)), ((n_neighbors+1)))
                # find kriging weights [https://en.wikipedia.org/wiki/Kriging#Ordinary_kriging]
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond=None) 

                est_ok[i] = local_mean + np.sum(k_weights[:n_neighbors]
                          * (vals_known[:] - local_mean)) 
                var_ok[i] = var_1 - np.sum(k_weights[:n_neighbors]
                          * covariance_array[:n_neighbors])
                var_ok[var_ok < 0] = 0

        return est_ok, var_ok
  
    @timeit
    def skrige_sgs(xyz_predict, df_known, num_points, vario, radius,
                   randseed=None, origin=None, quiet=False, desc=""):
        """
        Perform one Sequential Gaussian Simulation using Simple Kriging 
        
        Parameters
        ----------
            xyz_predict : numpy.ndarray
                x,y,z coordinates of prediction grid, or query points to be estimated
            df_known : pandas DataFrame
                with column names ['X','Y','Z'] for coordinates, 'V' for values
                data frame of conditioning data
            num_points : int
                the number of conditioning points to search for
            vario : list
                list of variogram parameters [nugget, range, sill, nu, vtype, R, S]
                nugget, range, sill are common to all vtypes, supported vtypes
                include {'Exponential', 'Spherical', 'Matern', 'Gaussian'}
                nu describes the smoothness of the Matern kernel (e.g. 1.5)
                R is the rotation matrix, either numpy.array of shape (3,3) or None
                S is the scaling array, either numpy.array of shape (3,) or None
            radius : int, float
                search radius
            randseed : int
                determines the random path used during sequential simulation
            origin : np.array of shape:(3,) or None
                ability to fix the origin when rotation is applied to a dataset
            quiet : bool
                If False, a progress bar will be printed to the console.
            desc : str (Optional)
                short description to include in execution time console output
        
        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in xyz_predict
        """

        # unpack variogram parameters
        nugget, effective_range, sill, nu, vtype, R, S = vario
        var_1 = sill

        path_indices = np.arange(len(xyz_predict))
        # set the seed for reproducible results
        if randseed is not None:
            np.random.seed(randseed)
        np.random.shuffle(path_indices)
        sgs = np.zeros(len(xyz_predict))
        df = pd.DataFrame(df_known)
        mean_1 = df['V'].mean() 

        for i in path_indices:
            xyz_star = xyz_predict[i]
            matches = np.sum(np.isclose(df[['X','Y','Z']].values, xyz_star), axis=1)==3
            if any(matches):
                sgs[i] = df['V'].values[np.where(matches)[0][0]]
            else:
                # find nearest neighbors
                nearest = NearestNeighbor.nearest_neighbor_search(
                          radius, num_points, xyz_star, df[['X','Y','Z','V']])
                xyz_known = nearest[:, :-1]
                vals_known = nearest[:,-1]
                # covariance between data
                covariance_matrix = Covariance.make_covariance_matrix(
                                    xyz_known, vario, origin)
                # covariance between data and unknown
                covariance_array = Covariance.make_covariance_array(
                                    xyz_known, xyz_star, vario, origin)
                # compute kriging weights
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix,
                                                          covariance_array, rcond = None)
                # get a stochastic estimate
                mu = mean_1 + np.sum(k_weights * (vals_known - mean_1)) 
                sigma2 = np.abs(var_1 - np.sum(k_weights * covariance_array))
                sgs[i] = np.random.normal(mu, np.sqrt(sigma2), 1)

            #augment estimate to conditional data
            df = pd.concat([df, pd.DataFrame({
                            'X': [xyz_star[0]],
                            'Y': [xyz_star[1]],
                            'Z': [xyz_star[2]],
                            'V': [sgs[i]]})], ignore_index=True)
        return sgs

    @timeit
    def okrige_sgs(xyz_predict, df_known, num_points, vario, radius,
                   randseed=None, origin=None, quiet=False, desc=""):
        """
        Perform one Sequential Gaussian Simulation using Ordinary Kriging 
        
        Parameters
        ----------
        Refer to `skrige_sgs`

        Returns
        -------
            sgs : numpy.ndarray
                simulated value for each coordinate in xyz_predict
        """

        # unpack variogram parameters
        nugget, effective_range, sill, nu, vtype, R, S = vario
        var_1 = sill

        path_indices = np.arange(len(xyz_predict))
        # set the seed for reproducible results
        if randseed is not None:
            np.random.seed(randseed)
        np.random.shuffle(path_indices)
        sgs = np.zeros(len(xyz_predict))
        df = pd.DataFrame(df_known)

        for i in path_indices:
            xyz_star = xyz_predict[i]
            matches = np.sum(np.isclose(df[['X','Y','Z']].values, xyz_star), axis=1)==3
            if any(matches):
                sgs[i] = df['V'].values[np.where(matches)[0][0]]
            else:
                # gather nearest neighbor points
                nearest = NearestNeighbor.nearest_neighbor_search(
                          radius, num_points, xyz_star, df[['X','Y','Z','V']])
                xyz_known = nearest[:, :-1]
                vals_known = nearest[:,-1]
                local_mean = np.mean(vals_known) 
                n_neighbors = len(nearest)
                # covariance between data
                covariance_matrix = np.zeros((n_neighbors+1, n_neighbors+1))
                covariance_matrix[:n_neighbors,:n_neighbors] = \
                    Covariance.make_covariance_matrix(xyz_known, vario)
                covariance_matrix[-1,:n_neighbors] = 1
                covariance_matrix[:n_neighbors,-1] = 1
                # set up RHS (covariance between data and unknown)
                covariance_array = np.zeros(n_neighbors+1)
                k_weights = np.zeros(n_neighbors+1)
                covariance_array[:n_neighbors] = \
                    Covariance.make_covariance_array(xyz_known, xyz_star, vario)
                covariance_array[n_neighbors] = 1 
                covariance_matrix.reshape(((n_neighbors+1)), ((n_neighbors+1)))
                # find kriging weights
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, 
                                                          covariance_array, rcond=None)
                mu = local_mean + np.sum(k_weights[:n_neighbors]
                   * (vals_known - local_mean))
                sigma2 = np.abs(var_1 - np.sum(k_weights[:n_neighbors]
                         * covariance_array[:n_neighbors]))
                # get a stochastic estimate
                sgs[i] = np.random.normal(mu, np.sqrt(sigma2), 1)

            #augment estimate to conditional data
            df = pd.concat([df, pd.DataFrame({
                            'X': [xyz_star[0]],
                            'Y': [xyz_star[1]],
                            'Z': [xyz_star[2]],
                            'V': [sgs[i]]})], ignore_index=True)
        return sgs

    def transform_coordinates(df_predict, df_known, cfg):
        """
        Compute the rotated and scaled coordinates in-place
        when cfg['kriging:transform_data'] is True (default is False)
        """
        transformed = False
        if cfg.get('kriging:transform_data', False):
            coords = ['X','Y','Z']
            xyz_predict_input = df_predict[coords].values
            xyz_known_input = df_known[coords].values
            # checks
            R = cfg.get('transformation:rotation_matrix', None)
            S = cfg.get('transformation:scaling_vector', None)
            if R is None:
                R = np.eye(3)
            assert(R.shape == (3,3) and np.isclose(np.linalg.det(R), 1))
            if S is None:
                S = np.ones(3)
            else:
                S = S / np.max(S) #scaling of axes is relative not absolute
            assert(S.shape == (3,))
            origin = cfg.get('transformation:origin', None)
            if origin is None:
                origin = np.mean(xyz_known_input, axis=0)
            # compute
            SR = np.diag(1./S) @ R
            df_predict.loc[:,coords] = np.matmul(xyz_predict_input - origin, SR.T)
            df_known.loc[:,coords] = np.matmul(xyz_known_input - origin, SR.T)
            transformed = True
            # feedback
            fmt = lambda vec: '[' + ','.join(['%.9g' % x for x in vec]) + ']'
            fmt2 = lambda mat: '[' + ','.join(['%s' % fmt(row) for row in mat]) + ']'
            print(f'Transformed input (coordinates - origin) with diag(1/S) @ R '
                  f'where R={fmt2(R)}, S={fmt(S)}, origin={fmt(origin)}')

        return df_predict, df_known, transformed

#======================================================================

class KrigingManager:
    """
    Provide an interface that coordinates multiple kriging sequential simulations.

    Note: Apart from invoking the SGS routine in a FOR loop, it also handles
    spatial transformation (rotation and scaling), normal score transformation,
    selection of unique model-period and domain dependent random seeds (this
    establishes deterministic random paths that lead to reproducible results),
    and fitting variogram to the training data (known samples in df_data).
    """

    def _kriging_common_workflow(method, df_train, df_predict, cfg):
        """
        Implements the general workflow for prediction using kriging techniques

        Parameters
        ----------
            method : str
                identifies one of the supported approaches ['SK', 'OK', 'SK-SGS', 'OK-SGS']
            df_train : pandas DataFrame
                corresponds to training data (some combination of exploration and blast holes)
                with column names ['X','Y','Z'] for coordinates, 'V' for values in specified domain
            df_predict : pandas DataFrame
                corresponds to predict locations (e.g. blocks from a domain in the bench below)
                with column names ['X','Y','Z'] for coordinates, and unknown values 'V'
            cfg: python dict
                specifies the parameters and chosen options for a given experiment
                'kriging:type': str
                    must be 'simple_kriging' or 'ordinary_kriging'
                'kriging:transform_data': bool
                    applies rotation and relative scaling to align samples with major
                    trends before the empirical variogram is fitted to the data
                'kriging:num_points' : int
                    the number of conditioning points used in kriging estimate,
                    this should be data-dependent (default: 16)
                'kriging:covariance_fn' : str
                    supported models include 'matern' (our default), 'exponential',
                    'gaussian', 'cubic', 'stable' and 'spherical' (scikit-gstat default)
                'kriging:matern_smoothness' : float or None
                    if None, nu parameter is determined during variogram estimation;
                    otherwise, it is bounded by the supplied value +/- 0.0001
                'kriging:hide_progress' : float
                    if False, progress bar will be printed to the console
                'kriging:apply_normal_score_transform': bool
                    if True, normal score transformation is applied to values in 'V'
                'transformation:normal_score_randseed' :
                    provides a reproducible random state for subsampling and noise-
                    smoothing in normal score transformation
                'transformation:rotation_matrix' : numpy.array of shape (3,3) or None
                    specifies a rotation matrix consistent with Vulcan conventions.
                    Rotation is always applied, either to align the data with known
                    geological trend upfront, OR as part of nearest neighbour search
                    during kriging regression. When `kriging:transform_data` is set to
                    True, it is applied to training (and query) data before the empirical
                    variogram is computed. In this case, rotation subsequently is disabled
                    during local neighbors search as directional dependencies have been
                    accounted for (both training and inference locations are aligned
                    with scaled axes). In the absence of scaling, rotation alone should
                    not affect the estimated parameters as the variogram is isotropic
                    (non-directional). The situation is different for GP, since kernel
                    length scales are estimated separately for the X, Y and Z axes.
                'transformation:scaling_vector' : numpy.array of shape (3,) or None
                    specifies a scaling consistent with the radii of the search ellipse.
                    Scaling always applies to nearest neighbour search during kriging
                    regression, in part because range-normalisation on the lag
                    parameter is required by all skgstat Variogram models. Additionally,
                    relative scaling is applied to the training data before the empirical
                    variogram is computed if `kriging:transform_data` is set to True.
                    This matters as stretching by max(sX,sY,sZ)/[sX,sY,sZ] in the rotated
                    frame may affect the effective range reported by the variogram.
                'transformation:origin' : numpy.array of shape (3,3) or None
                    specifies the origin for data transformation
                'variogram:max_lag' : float
                    maximum distance used in the computation of the empirical variogram,
                    note: if the vulcan search ellipse dims are [500,450,300] say,
                    these being half the x,y,z ranges, then, 1000 should be used.
                'variogram:num_lag' : int
                    number of lag intervals defined by the binning function
                'variogram:use_nugget' : bool
                    whether to model nugget effect (default: False)
                'simulation:result_dir' : str
                    if non-empty, the values from each simulation will be flushed
                    incrementally and written to a file
                'simulation:num' : int
                    total number of sequential simulations
                'info:period_id' : int
                    time period identifier, the mA part in "mA_mB_mC" where for instance
                    mA = 4 implies we are inferring values using causal data for
                    the operation period (weeks) denoted "4_5_6"
                'info:domain_id' : int
                    four digit domain id "<limbzone><gradezone><porferyzone><rocktype>"
                'info:block_columns_to_copy' : str
                    columns to copy from blocks_to_estimate.csv into the simulation results file
        """
        if method not in ['SK', 'OK', 'SK-SGS', 'OK-SGS']:
            raise NotImplementedError(f"{method} is unknown or unsupported")

        bl, tr = KrigingManager._get_region_bounds(df_predict, cfg)
        df_known = KrigingManager._region_clip(df_train, bl, tr).copy()
        df_infer = df_predict.copy()

        # spatial transformation
        # - rotation/scaling procedure is only applied if instructed by the cfg
        # - the returned variable `transformed` indicates if this was applied
        R = cfg.get('transformation:rotation_matrix', None)
        S = cfg.get('transformation:scaling_vector', None)
        df_infer, df_known, transformed = KrigingRegression.transform_coordinates(df_infer, df_known, cfg)
        # - if the known data and predict are transformed here, during inference,
        #   local neighbourhood search does not require further transformation
        if transformed:
            R, S = None, None

        # normal score transformation
        nst_preference = cfg.get('kriging:apply_normal_score_transform', False)
        apply_normal_score_xform = True if 'SGS' in method else nst_preference
        if len(df_known) < cfg.get('transformation:nst_min_sample_size', 30):
            apply_normal_score_xform = False

        if apply_normal_score_xform:
            df_known = df_known.rename(columns = {'V':'V_input'})
            raw_vals = df_known['V_input'].values.reshape(-1,1)
            rstate = cfg.get('transformation:normal_score_randseed', 8725)
            nst_trans = QuantileTransformer(n_quantiles=min(len(df_known), 1000),
                        output_distribution='normal', random_state=rstate).fit(raw_vals)
            df_known.loc[:,'V'] = nst_trans.transform(raw_vals)
        # assign values to transformed_vals as df_known might be subsequently downsampled
        transformed_vals = df_known['V'].values

        # subsample training samples to manage computation load
        df_known = KrigingManager._limit_training_samples(df_known, cfg)

        # experiment description
        domain_id = int(cfg.get('info:domain_id', 8888))
        domain_key = domain_id_to_column_name(domain_id)
        mA = int(cfg.get('info:period_id', 1))
        inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_serial = KrigingManager._make_experiment_serial(method, cfg)
        cfg['internal:experiment_serial'] = experiment_serial

        # compute variogram
        variogram_params = KrigingManager._read_parameters_from_csv(cfg)
        variogram_model = cfg.get('kriging:covariance_fn', 'matern')
        t0 = time.time()
        if variogram_params is None:
            vgr = KrigingManager.compute_variogram(df_known, cfg)
            # stop further processing if a valid variogram cannot be obtained
            if vgr is None:
                cfg['variogram:params'] = [np.nan] * 4 + [variogram_model, R, S]
                cfg['timing:learn'] = np.nan
                cfg['timing:inference'] = np.nan
                if 'SGS' in method:
                    num_simulations = int(cfg.get('simulation:num', 20))
                    r = re.compile(r'(\w)[a-z]*_(\w)[a-z]*')
                    t = r.match(cfg['kriging:type'])
                    sim_column = '{}{}_SGS_'.format(t[1].upper(),t[2].upper())
                    cfg['simulation:column_names'] = []
                    for i in np.arange(num_simulations):
                        col_name = sim_column + str(i)
                        cfg['simulation:column_names'].append(col_name)
                        df_infer.loc[:,col_name] = np.nan * np.ones(len(df_infer))
                    return df_infer[cfg['simulation:column_names']] if not cfg.get('bypass_simulation', False) else None
                else:
                    estim_mean = np.nan * np.ones(len(df_infer))
                    estim_var = np.nan * np.ones(len(df_infer))
                    return estim_mean, estim_var

            variogram_params = vgr.parameters
            learning_space = KrigingManager._learning_space_designation(cfg)
            hyperparams = defaultdict(list)
            hyperparams['timestamp'].append(timestamp)
            hyperparams['learning_space'].append(learning_space)
            hyperparams['variogram_model'].append(variogram_model)
            hyperparams['inference_period'].append(inference_period)
            hyperparams['domain'].append(domain_id)
            hyperparams['experiment_serial'].append(experiment_serial)
            # The params [range, sill, <nu>, nugget] written to file always
            # include the optional shape parameter nu which is only present
            # in vgr.parameters when a "matern" variogram model is used.
            # So, we splice this and insert "nu" between [range, sill] and
            # [nugget] if necessary to maintain consistency in the csv file.
            written_params = variogram_params
            if variogram_model != 'matern':
                written_params = variogram_params[:-1] + [np.nan, variogram_params[-1]]
            hyperparams['variogram_params'].append(written_params)
            KrigingManager._write_parameters_to_csv(hyperparams, cfg)

        # - extract variogram parameters
        range, sill = variogram_params[:2]
        nu = variogram_params[-2] if variogram_model == 'matern' else np.nan
        nugget = variogram_params[-1]
        search_points = cfg.get('kriging:num_points', 16)
        search_radius = 0.5 * min(cfg.get('variogram:max_lag', 1000), range)
        # - local geometry transformation (if any) is described by R and S
        cfg['variogram:params'] = variogram = [nugget, range, sill, nu, variogram_model, R, S]
        cfg['timing:learn'] = time.time() - t0

        # invoke relevant method
        if 'SGS' in method:
            # generate time-period and domain specific random seeds to
            # produce unique and reproducible random paths
            cfg['simulation:period_domain_id'] = s = f"{inference_period}:{domain_key}"
            cfg['simulation:period_domain_initial_state'] = \
                int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
            rng = np.random.default_rng(seed=cfg['simulation:period_domain_initial_state'])
            num_simulations = int(cfg.get('simulation:num', 20))
            cfg['simulation:path_seeds'] = rng.integers(low=0, high=2**31, size=num_simulations)

            # sequential simulation loop
            cfg['simulation:column_names'] = []
            # for inference, a random path is chosen internally given a random seed
            xyz_predict = df_infer[['X','Y','Z']].values
            r = re.compile(r'(\w)[a-z]*_(\w)[a-z]*')
            t = r.match(cfg['kriging:type'])
            sim_column = '{}{}_SGS_'.format(t[1].upper(),t[2].upper())
            quiet = cfg.get('kriging:hide_progress', True)
            t0 = time.time()
            for i in np.arange(num_simulations):
                col_name = sim_column + str(i)
                cfg['simulation:column_names'].append(col_name)
                if not cfg.get('bypass_simulation', False):
                    if method == 'OK-SGS':
                        sim = KrigingRegression.okrige_sgs(xyz_predict, df_known, search_points, variogram,
                              search_radius, cfg['simulation:path_seeds'][i], quiet=quiet, desc=str(i))
                    elif method == 'SK-SGS':
                        sim = KrigingRegression.skrige_sgs(xyz_predict, df_known, search_points, variogram,
                              search_radius, cfg['simulation:path_seeds'][i], quiet=quiet, desc=str(i))
                    # - reverse normal score transformation
                    if apply_normal_score_xform:
                        v_star = nst_trans.inverse_transform(sim.reshape(-1,1))
                    else:
                        v_star = sim
                    df_infer.loc[:,col_name] = v_star

            cfg['timing:inference'] = time.time() - t0
            if not cfg.get('bypass_simulation', False):
                return df_infer[cfg['simulation:column_names']]

        else: # no sequential simulation
            quiet = True
            xyz_predict = df_infer[['X','Y','Z']].values
            num_points = cfg['kriging:num_points']
            t0 = time.time()
            if method == 'SK': #simple kriging
                estim_mean, estim_var = KrigingRegression.skrige(
                                        xyz_predict, df_known, num_points,
                                        variogram, search_radius, quiet)
            elif method == 'OK': #ordinary kriging
                estim_mean, estim_var = KrigingRegression.okrige(
                                        xyz_predict, df_known, num_points,
                                        variogram, search_radius, quiet)
            else:
                raise NotImplementedError(f"{cfg['kriging:type']} is unknown or unsupported")
            cfg['timing:inference'] = time.time() - t0

            if apply_normal_score_xform:
                estim_mean = nst_trans.inverse_transform(estim_mean.reshape(-1,1)).flatten()
                # estimate variance using Taylor series approximation
                # V[f(X)] = f'(mu)[f'(mu) - mu*f"(mu)]*sigma^2 + ...... ~= f'(mu)^2 * V[X]
                eps = cfg.get('kriging:nst_stdev_correction_epsilon', 1e-9)
                raw_vals_x = raw_vals.flatten()
                transformed_vals_fx = transformed_vals.flatten()
                iSort = np.argsort(raw_vals_x)
                gradients = np.diff(transformed_vals_fx[iSort]) / (eps + np.diff(raw_vals_x[iSort]))
                gradients_trunc = np.sort(gradients[gradients > 0])
                n = len(gradients_trunc)
                m = 0.5 * (n - 1)
                f_prime_estim = np.median(gradients_trunc)
                f_inverse_std = np.sqrt(np.maximum(estim_var, 0)) / f_prime_estim
                estim_var = f_inverse_std**2

            return estim_mean, estim_var

    def kriging_sequential_simulations(df_known, df_predict, cfg):
        """
        API to perform multiple Sequential Gaussian Simulations using Simple or Ordinary Kriging

        Parameters
        ----------
        Refer to `_kriging_common_workflow`

        Effects
        -------
        1)  df_infer (represents a DataFrame with indices corresponding to a specific domain)
            has columns "<t>_SGS_<i>" that contain simulated values.
            <t> denotes the kriging type such as OK or SK for ordinary or simple kriging
            <i> denotes iteration i in the sequential simulation.
        2)  cfg['variogram:params'] will be populated with [nugget, range, sill, <shape(nu)>,
            variogram_model, R, S] where R and S are the local rotation and scaling matrices
        3)  cfg['simulation:column_names'] will contain a list of column names for simulated values
        4)  random states and identifiers will be added to the cfg, in the form of
            ['simulation:period_domain_id'] that resembles "{inference_period}:{domain_key}",
            ['simulation:period_domain_initial_state']: a SHA256 modulo 10**8 hash value,
            ['simulation:path_seeds']: list of M random seeds chosen for each simulation.
        """
        method = 'OK-SGS' if cfg['kriging:type'] == 'ordinary_kriging' else 'SK-SGS'
        return KrigingManager._kriging_common_workflow(method, df_known, df_predict, cfg)

    def kriging_regression(df_known, df_predict, cfg):
        """
        API to perform ordinary kriging or simple kriging (without sequential simulation)

        Parameters
        ----------
        Refer to `_kriging_common_workflow`
        """
        method = 'SK' if cfg['kriging:type'] == 'simple_kriging' else 'OK'
        return KrigingManager._kriging_common_workflow(method, df_known, df_predict, cfg)

    def compute_variogram(df_data, cfg):
        # compute experimental (isotropic) variogram
        fixed_nu = cfg.get('kriging:matern_smoothness', None)
        max_range = cfg.get('variogram:max_lag', 1000)
        constraints = None
        # - option to constrain the `nu` smoothness parameter for Matern, specifying
        #   (lower, upper) bounds on params passed to scipy.optimize.curve_fit
        #   upper = [max_range, max_sill, max_nu, max_nugget], similar for lower.
        # - None means default internal constraints will be used, nu will be variable.
        #   For Matern 3/2, nu=1.5 (Note: nu=0.5 is exponential, nu>=20 is Gaussian)
        if fixed_nu:
            max_var = np.var(df_data['V'].values)
            constraints = ([0., 0., fixed_nu - 0.0001, 0.],
                          [max_range, max_var, fixed_nu + 0.0001, 0.5 * max_var])
            if cfg['variogram:use_nugget'] is False:
                constraints = tuple([cs[:-1] for cs in constraints])

        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='All input values are the same*')
            try:
                vgr = skg.Variogram(coordinates=df_data[['X','Y','Z']].values,
                          values=df_data['V'].values,
                          estimator=cfg.get('variogram:estimation_method', 'matheron'),
                          model=cfg.get('kriging:covariance_fn', 'matern'),
                          use_nugget=cfg.get('variogram:use_nugget', False),
                          maxlag=max_range,
                          n_lags=cfg.get('variogram:num_lag', 80),
                          fit_bounds=constraints)
            except Warning as e:
                print('Warning: variogram result is invalid. Reason:', e)
                vgr = None

        return vgr #note: vgr.parameters gives [range, sill, <shape>, nugget]

    def _get_region_bounds(df_infer, cfg):
        margin = cfg.get('data:training_region_margin', [0,0,0])
        bl = df_infer.loc[:, ['X','Y','Z']].min().values - margin
        tr = df_infer.loc[:, ['X','Y','Z']].max().values + margin
        return bl, tr

    def _region_clip(df, bl, tr):
        xyz = df.loc[:, ['X','Y','Z']].values
        idx_in_bounds = np.all((xyz > bl[np.newaxis,:]) & (xyz < tr[np.newaxis,:]), axis=1)
        return df.loc[idx_in_bounds]

    def _remove_chars(s, chars):
        for c in chars:
            s = s.replace(c, '')
        return s

    def _make_experiment_serial(method, cfg):
        # Return a descriptive string of the form <method>[r]_bt_bi
        # where <method> = is one of {"SK", "SK-SGS" "OK", "OK-SGS"}
        # [r] is included when cfg['kriging:transform_data'] is True
        prefix = cfg.get('kriging:rotation_serial_prefix', '')
        if len(prefix) > 0:
            prefix += '_'
        method_ = KrigingManager._remove_chars(method.lower(), ['-','(',')'])
        r = 'r' if cfg.get('kriging:transform_data', False) else ''
        transform = "_nst" if cfg.get('kriging:apply_normal_score_transform', False) else ""
        return f"{prefix}{method_}{r}_bt_bi{transform}"

    def _format_variogram_params(x):
        re_cascade = lambda y, pA, pB, pC: re.sub(pC, '', re.sub(pB, '', re.sub(pA, ' ', y)))
        remove_enclosed_typename = lambda y: re.sub(r'np.float\d*\(', '', re.sub(r'\)', '', y))
        scientific_to_float = lambda y: ' '.join(['%.12g' % float(i) for i in re.split(r',\s*| ', y)])
        return '[' + scientific_to_float(remove_enclosed_typename(
                     re_cascade(str(x), r'\s+', r'\s*\[\s*', r'\s*\]\s*'))) + ']'

    def _learning_space_designation(cfg):
        return 'rotated' if cfg.get('kriging:transform_data', True) else 'not_rotated'

    def _write_parameters_to_csv(hyperparams_dict, cfg):
        csvname = cfg.get('kriging:hyperparams_csv_file', 'gstatsim3d_optimised_parameters_kriging.csv')
        fname = Path(cfg['info:data_dir']) / csvname
        df = pd.DataFrame(hyperparams_dict)
        df['variogram_params'] = df['variogram_params'].apply(KrigingManager._format_variogram_params)
        # variogram_params contains [range, sill, <shape>, nugget]
        if os.path.isfile(fname):
            # check if entry with same attributes already exists
            empty_file = os.stat(fname).st_size == 0
            if not empty_file:
                dfr = pd.read_csv(fname, header=0, sep=',')
                mA = int(cfg.get('info:period_id', 1))
                learning_space = KrigingManager._learning_space_designation(cfg)
                variogram_model = cfg.get('kriging:covariance_fn', 'matern')
                inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
                domain_id = int(cfg['info:domain_id'])
                experiment_serial = cfg['internal:experiment_serial']
                matches = dfr.loc[(dfr['learning_space']==learning_space) &
                                  (dfr['variogram_model']==variogram_model) &
                                  (dfr['inference_period']==inference_period) &
                                  (dfr['experiment_serial']==experiment_serial) &
                                  (dfr['domain']==domain_id)].index
            if empty_file or len(matches) == 0:
                incl_hdr = True if empty_file else False
                df.to_csv(fname, index=False, float_format='%.9g', header=incl_hdr, mode='a')
        else:
            df.to_csv(fname, index=False, float_format='%.9g', header=True, mode='w')

    def _read_parameters_from_csv(cfg):
        csvname = cfg.get('kriging:hyperparams_csv_file', 'gstatsim3d_optimised_parameters_kriging.csv')
        fname = Path(cfg['info:data_dir']) / csvname
        parms = None
        try: #retrieve variogram params for specific conditions
            df = pd.read_csv(fname, header=0, sep=',')
            mA = int(cfg.get('info:period_id', 1))
            learning_space = KrigingManager._learning_space_designation(cfg)
            variogram_model = cfg.get('kriging:covariance_fn', 'matern')
            inference_period = "%02d_%02d_%02d" % (mA, mA+1, mA+2)
            domain_id = int(cfg['info:domain_id'])
            experiment_serial = cfg['internal:experiment_serial']
            row = df.loc[(df['learning_space']==learning_space) &
                         (df['variogram_model']==variogram_model) &
                         (df['inference_period']==inference_period) &
                         (df['experiment_serial']==experiment_serial) &
                         (df['domain']==domain_id)].index
            strip = lambda x: re.sub(r'\s*\]\s*', '', re.sub(r'\s*\[\s*', '', re.sub(r'\s+', ' ', x)))
            parms = [float(x) for x in strip(df.loc[row[0],'variogram_params']).split()]
            #remove the (2nd last) shape parameter "nu" if it is unused
            if variogram_model != 'matern':
                parms = parms[:-2] + [parms[-1]]
        except:
            pass
        return parms

    def _limit_training_samples(df_data, cfg):
        npts_before = len(df_data)
        if cfg.get('data:downsampling_mask', None) is None:
            x = df_data.loc[:, ['X','Y','Z']].values
            y = df_data.loc[:, 'V'].values
            retain = downsample(x, y, cfg.get('kriging:training_max_bh_points', 0))
            if len(retain) < npts_before:
                mask = np.zeros(len(y), dtype=bool)
                mask[retain] = 1
                cfg['data:downsampling_mask'] = mask
        else:
            retain = np.where(cfg['data:downsampling_mask'] > 0)[0]
        if len(retain) < npts_before:
            df_data = df_data.iloc[retain]
            print(f"Downsampled training data from {npts_before} to {len(retain)}")
        return df_data

__all__ = ['KrigingRegression', 'KrigingManager']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]
