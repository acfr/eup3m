# Compute rotation matrix "R" and scaling matrix "S" for each domain
# using search ellipse dimensions and orientation information (azimuth,
# plunge and dip) contained in cfg['gp:ellipse_definitions_filename']
#
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Raymond Leung and Alexander Lowe <raymond.leung@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------


from collections import defaultdict
from pathlib import Path

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


DOMAIN_ORIENTATION_DF = None


def get_domains_dataframe(cfg):
    """
    Retrieve ellipsoid orientation and scale data from csv file.
    The csv file is quite messy and requires a bit of cleaning up.

    Returns
    -------
        df : pandas.DataFrame
             with columns 'variable' and various domain_keys 'LZ{limbzone}_{gradezone}_{porphyryzone}_{rocktype}'
             the rows describe attributes "cudom", "lz", "vmodel", "angle1", "angle2", "angle3",
             "axis1", "axis2", "axis3", "c0", "sill_str1", "range1_str1", "range2_str1", "range3_str1",
             "type_str1", "sill_str2", "range1_str2", "range2_str2", "range3_str2", "type_str2",
             "search1x", "search1y", "search1z"
    """
    global DOMAIN_ORIENTATION_DF

    if DOMAIN_ORIENTATION_DF is None:
        filename = Path(cfg['info:data_dir']) / cfg.get('gp:ellipse_definitions_filename', 'future bench variograms.txt')
        df = pd.read_csv(filename, delimiter=',|=', skip_blank_lines=True, comment='#', engine='python',
                         converters=defaultdict(lambda: str.strip))
        df = df.rename(columns={c:c.strip() for c in df.columns}).transpose()
        df = df.rename(columns=df.iloc[0].apply(lambda s: s.replace("%VAR ", "").strip())).iloc[1:]
        DOMAIN_ORIENTATION_DF = df

    else:
        df = DOMAIN_ORIENTATION_DF

    return df


def get_unique_domains(cfg):
    domain_geom_file = Path(cfg['info:data_dir']) / cfg.get(
                       'info:domain_geometry_file', 'domain_ellipse_geometry.csv')
    if os.path.exists(domain_geom_file):
        df = pd.read_csv(domain_geom_file, index_col=0)
    else:
        df = get_domains_dataframe(cfg)
    return np.unique([int(domain_str[2:].replace('_', '')) for domain_str in df.index])


def get_domain_orientation(domain_id, cfg):
    """
    Retrieve orientation and radii of ellipses for the given domain
    :param domain_id: integer with 4 digits consisting of <limbzone><gradezone><porferyzone><rocktype>
    :param cfg: dict with default values for ellipse rotation and ellipse radius
    :return: (azimuth, plunge, dip, scaleX, scaleY, scaleZ)
    """
    domain_geom_file = Path(cfg['info:data_dir']) / cfg.get(
                       'info:domain_geometry_file', 'domain_ellipse_geometry.csv')
    if os.path.exists(domain_geom_file):
        df = pd.read_csv(domain_geom_file, index_col=0)
    else:
        df = get_domains_dataframe(cfg)

    lz = domain_id // 1000
    gz = (domain_id % 1000) // 100
    por = (domain_id % 100) // 10
    rock_type = domain_id % 10
    key = f"LZ{lz}_{gz}_{por}_{rock_type}"
    if key in df.index:
        row = df.loc[key, ['angle1', 'angle2', 'angle3', 'search1x', 'search1y', 'search1z']].astype(float)
        result = np.deg2rad(row['angle1']), np.deg2rad(row['angle2']), np.deg2rad(row['angle3']),\
                 row['search1x'], row['search1y'], row['search1z']
        return result

    else: #added extra outer parenthesis as tuple unpacking causes invalid syntax in python<=3.7
        return \
            (*cfg.get('gp:ellipse_default_rotation', (0., 0., 0.)),\
             *cfg.get('gp:ellipse_default_radius', (300., 300., 300.)))


def compute_rotation_and_scaling_matrix_vulcan(cfg, domain_id):
    """
    Interpretation:
        Azimuth/angle1 = angle of rotation (counter-clockwise) about the z-axis
        Plunge/angle2 = angle of rotation (clockwise) about the x-axis
        Dip/angle3 = angle of rotation (counter-clockwise) about the y-axis

    :brief Rotation and Scaling as specified by get_domain_orientation to match vulcan conventions
           Its behaviour is identical to calling `compute_any_rotation_and_scaling_matrix` with
           either `legacy_mode` or cfg['gp:associate_plunge_rotation_with_x'] set to True
    :param cfg:
    :param domain_id: integer with 4 digits consisting of <limbzone><gradezone><porferyzone><rocktype>
    :return: R, S.  Rotation matrix R such that b = R @ e where e are ellipse major,semi,minor vectors
                    and b are coordinate basis vectors.
                    S vector of ellipse radii major, semi, minor such that np.diag(1./S) @ R @ ellipse is a sphere of unit radius
    """
    return compute_any_rotation_and_scaling_matrix(cfg, domain_id, legacy_mode=True)


def compute_any_rotation_and_scaling_matrix(cfg, domain_id, legacy_mode=False):
    """
    Perform any of the 48 rotation permutation sequences using cfg['gp:rotation_sequence']
    Note: This method represents a generalisation of `compute_rotation_and_scaling_matrix_vulcan`
          and it differs in one important respect. According to Maptek/Vulcan's PythonSDK
          documentation, for ellipsoid modelling, "plunge" is associated with clockwise
          rotation about the Y axis (which points East) and "dip" is associated with
          clockwise rotation about the -X axis (which points North). This amounts to a
          counter-clockwise rotation about the X axis (which points South) in standard
          Cartesian coordinates. So, we associate "dip" with X, and "plunge" with Y unless
          `legacy_mode` is True OR cfg['gp:associate_plunge_rotation_with_x'] is True.
    """
    azimuth, plunge, dip, dimX, dimY, dimZ = get_domain_orientation(domain_id, cfg)
    theta = azimuth
    if legacy_mode or cfg.get('gp:associate_plunge_rotation_with_x', True):
        angleX, angleY = plunge, dip
        default_rotation_sequence = [1,3,2] # Ry @ Rx.T @ Rz
    else: #current interpretaion
        angleX, angleY = dip, plunge
        default_rotation_sequence = [0,1,5] # Rx @ Ry @ Rz.T
        
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]])
    Rx = np.array([
        [1., 0., 0.],
        [0., np.cos(angleX), -np.sin(angleX)],
        [0., np.sin(angleX), np.cos(angleX)]])
    Ry = np.array([
        [np.cos(angleY), 0., np.sin(angleY)],
        [0., 1., 0.],
        [-np.sin(angleY), 0., np.cos(angleY)]])

    rotation_matrices = {0: Rx, 1: Ry, 2: Rz, 3: Rx.T, 4: Ry.T, 5: Rz.T}
    # The default (experimentally verified) rotation system is Rx @ Ry @ Rz.T
    # The list in `spec` follows the same order as written in standard algebra
    spec = cfg.get('gp:rotation_sequence', default_rotation_sequence)
    R = rotation_matrices[spec[0]] @ rotation_matrices[spec[1]] @ rotation_matrices[spec[2]]
    S = np.array([dimX, dimY, dimZ])
    return R, S

def get_transformed_coordinates(xyz, scale_enable, rotation_enable, origin, domain_id, cfg,
                                get_rotation_and_scaling_matrix=compute_any_rotation_and_scaling_matrix):
    """
    transform points xyz such that if they formed an ellipse with radii and orientation as for the associated domain,
    then the transformed points would be the same ellipse, but with major,semi and minor axes-aligned with coordinate axes.
    :param xyz:
    :param scale_enable: if enabled, the ellipse would be collapsed to a unit sphere.
    :param rotation_enable: if not enabled, no axis alignment is performend.
    :param origin: centre of rotation
    :param domain_id: integer with 4 digits consisting of <limbzone><gradezone><porferyzone><rocktype>
    :param cfg: must have 'info:data_dir' key, under which the file "future bench variograms.txt" can be found
    :return:
    """
    if not scale_enable and not rotation_enable:
        return xyz

    R, S = get_rotation_and_scaling_matrix(cfg, domain_id)
    SR = np.eye(3, dtype=float)
    if rotation_enable:
        SR = R
    if scale_enable:
        SR = np.diag(1./S) @ SR

    origin = np.array(origin)
    xyz = (xyz - origin[np.newaxis, :]) @ SR.T
    return xyz
