# Default configuration and miscellaneous functions for display/data manipulation
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#-------------------------------------------------------------------------------


import matplotlib as mpl
import numpy as np

# Kriging config parameters are described in the
# KrigingManager._kriging_common_workflow doc-string
def create_kriging_config(R=None, S=None, specs={}):
    cfg = {
        'kriging:type': 'ordinary_kriging',
        'kriging:transform_data': True,
        'kriging:num_points' : 9,
        'kriging:covariance_fn': 'matern',
        'kriging:matern_smoothness': None,
        'kriging:training_max_bh_points' : 3000,
        'kriging:hide_progress': True,
        'transformation:normal_score_randseed': 8725,
        'transformation:rotation_matrix': R,
        'transformation:scaling_vector': S,
        'transformation:origin': None,
        'variogram:max_lag' : 1000.0,
        'variogram:use_nugget': True,
        'simulation:result_dir': None,
        'simulation:num' : 4,
        'data:training_region_margin': [3000., 3000., 3000.],
        'data:downsampling_mask': None,
        'info:data_dir' : specs.get('info:data_dir', 'data'),
        'info:period_id': specs.get('mA', 4),
        'info:domain_id': specs.get('domain_id', 2310)
    }
    # Override default parameters
    for k, v in specs.items():
        if k in cfg:
            cfg[k] = v
    return cfg

# Gaussian process config parameters are described in the
# GPManager.gaussian_process_simulations doc-string
def create_gaussian_process_config(specs={}):
    cfg_base = {
        'gp:inference_datasets' : ["blastholes", "exploration"],
        'gp:volumetric': False,
        'gp:exp_height': 0.,
        'gp:blasthole_height' : 50., #(ft) unused
        'gp:training_max_bh_points' : 500,
        'gp:training_max_exp_points' : 250,
        'gp:training_max_all_points' : 1000000,
        'gp:training_region_margin': [3000., 3000., 3000.],
        'gp:training_local_neighborhood_enable' : False,
        'gp:inference_local_neighborhood_enable' : True,
        'gp:local_neighborhood_ellipse_knn_min_max': (2, 9),
        'gp:local_neighborhood_scale_enable' : True,
        'gp:local_neighborhood_rotation_enable' : True,
        'gp:learning_inference_in_rotated_space' : False,
        'gp:ellipse_definitions_filename' : "future bench variograms.txt",
        'gp:ellipse_default_radius' : (300., 300., 300.),
        'gp:ellipse_default_rotation' : (0., 0., 0.),
        'gp:kernel_name' : 'Matern32',
        'gp:init_hyperparams' : {},
        'gp:fixed_hyperparams' : {},
        'gp:length_scale_bounds' : [25., 1000.],
        'gp:max_repeats' : 3,
        'gp:init_state': 2633,
        'gp:mean_col_name' : None,
        'gp:stdev_col_name' : None,
        'gp:log_nlml_r2_with_hyperparams': True,
        'gp:log_file' : "gstatsim3d_gp.log",
        'gp:clear_previous_log': True,
        'domain_column_name' : 'lookup_domain',
        'simulation:filename_template' : 'gstatsim3d_gp.csv',
        'simulation:num' : 4,
        'data:downsampling_mask': None,
        'info:data_dir' : specs.get('info:data_dir', 'data'),
        'info:period_id' : None,
        'info:domain_id' : None
    }
    cfg_override = {
        'gp:mean_col_name' : "mean_GPR_gbt_lbi",
        'gp:stdev_col_name' : "stdev_GPR_gbt_lbi",
        'gp:inference_datasets': ["blastholes"],
        'gp:training_max_bh_points' : 3000,
        'gp:training_max_exp_points' : 0,
        'gp:training_local_neighborhood_enable': False,
        'gp:inference_local_neighborhood_enable': True,
        'gp:learning_inference_in_rotated_space': True,
        'gp:associate_plunge_rotation_with_x': False,
        'gp:nst_stdev_correction_enable': True,
        'info:period_id' : specs.get('mA', 4),
        'info:domain_id' : specs.get('domain_id', 2310)
    }
    cfg_base.update(cfg_override)
    for k, v in specs.items():
        if k in cfg_base:
            cfg_base[k] = v
    return cfg_base

def create_inverted_colormap(gamma=1, levels=256, monochrome=False):
    """
    Combine a symmetric blue/red colour scale to cover range [-1,1] with darker shades near zero
    :param gamma: nonlinearity (< 1 to expand, > 1 to compress small values nearest to zero)
    """
    x = np.linspace(0, 1, levels//2)
    xp = x**gamma
    cmap_upper = mpl.cm.Reds(np.linspace(0, 1, levels//2))
    cmap_lower = mpl.cm.Blues(np.linspace(0, 1, levels//2))
    lower, upper = np.zeros((levels//2, 4)), np.zeros((levels//2, 4))
    for rgba in range(4):
        upper[:,rgba] = np.interp(xp, x, cmap_upper[:,rgba])[::-1]
        lower[:,rgba] = np.interp(xp, x, cmap_lower[:,rgba])
    if monochrome:
        cmap = mpl.colors.ListedColormap(np.r_[lower])
    else:
        cmap = mpl.colors.ListedColormap(np.r_[lower, upper])
    return cmap

def variograms_colour_lookup(word):
    """
    Map colour word to hex/str. Help define a consistent colour scheme
    """
    lut = {'blue-dark':   'b',          #OK
           'magenta':     'm',          #OK_SGS_single
           'blue-light':  'tab:blue',   #OK_SGS_mean_from
           'green-dark':  '#006600',    #SK
           'olive':       '#77933C',    #SK_SGS_single
           'green-mid':   '#009900',    #SK_SGS_mean_from
           'lilac':       'tab:purple', #GP(L)
           'red-dark':    '#990033',    #GP_SGS_single
           'red':         'tab:red',    #GP_SGS_mean_from
           'pink':        'tab:pink',   #GP(G)
           'orange-dark': '#CC5300',    #CP_CRF_single
           'orange':      'tab:orange', #CP_CRF_single
           'gray':        'tab:gray',   #RTK_LongRangeModel
           'brown':       'tab:brown',  #RTK_BenchAboveData
           'black':       'k'           #GroundTruth
    }
    return lut[word]

def check_learning_rotation_status(specs):
    # Check if variogram fitting and GP learning occur in rotated space
    learning_in_rotated_space = {
        'kriging': specs.get('kriging:transform_data', True),
        'gp': specs.get('gp:learning_inference_in_rotated_space', True)
    }
    if learning_in_rotated_space['kriging']:
        assert(learning_in_rotated_space['gp'])
    else:
        assert(not learning_in_rotated_space['gp'])
    return 'learning_rotated' if learning_in_rotated_space['gp'] else 'not_rotated'

def extract_raw_nst_predictions(df, return_variance=False):
    # Factored out common data extraction steps from a pandas.DataFrame.
    # Note: For consistency, we always write the standard deviation to csv files.
    #       However, the kriging code we inherited returns the variance instead.
    #       We choose not to alter these APIs and use the `return_variance` flag
    #       to manage these differences and preserve the existing behaviour.
    moment_1st, col_1st = {}, {}
    moment_2nd, col_2nd = {}, {}
    col_1st['raw'] = [x for x in df.columns if 'mean' in x and 'nst' not in x][0]
    col_1st['nst'] = [x for x in df.columns if 'mean' in x and 'nst' in x][0]
    col_2nd['raw'] = [x for x in df.columns if 'stdev' in x and 'nst' not in x][0]
    col_2nd['nst'] = [x for x in df.columns if 'stdev' in x and 'nst' in x][0]
    moment_1st['raw'] = df[col_1st['raw']].values
    moment_1st['nst'] = df[col_1st['nst']].values
    x2_raw = df[col_2nd['raw']].values
    x2_nst = df[col_2nd['nst']].values
    moment_2nd['raw'] = x2_raw**2 if return_variance else x2_raw
    moment_2nd['nst'] = x2_nst**2 if return_variance else x2_nst
    return moment_1st, moment_2nd, col_1st, col_2nd
