# Implement neighbourhood search using scipy.spatial.CKDTree
# 
# Rio Tinto Centre
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2024 Alexander Lowe <alexander.lowe@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#-------------------------------------------------------------------------------


import numpy as np
from scipy.spatial import cKDTree


def find_neighbours(xyz, xyz_queries, radius=None, knn_min=2, knn_max=9, tree=None, p_distance_exponent=2):
    """
    :brief find the closest knn_min points and those within radius distance up to maximum of knn_max
    :param xyz: numpy array of points
    :param tree: scipy.spatial cKDTree
    :param xyz_queries: nx3 array
    :param radius: 3-tuple
    :param knn_min: int
    :param knn_max: int
    :return: dist, idx_near, tree where ix are indices into xyz
    """
    if len(xyz) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), tree

    knn_min, knn_max = int(knn_min), int(knn_max)

    if radius is not None:
        radius = np.array(radius).flatten()
        if len(radius) == 1:
            radius = np.full(xyz.shape[1], radius[0])
        xyz = xyz / radius[np.newaxis, :]
        xyz_queries = xyz_queries / radius[np.newaxis, :]

    if tree is None:
        tree = cKDTree(xyz)

    d, idx_near = tree.query(xyz_queries, knn_max, p=p_distance_exponent)
    if radius is not None:
        column_number = np.array(len(d) * [[c for c in range(knn_max)]])
        mask = (column_number > knn_min) & (d > 1.)
        idx_near[mask] = len(xyz)

    return d, idx_near, tree


def octant_search(xyz, xyz_query, plentiful_idx, knn_max, debug=False):
    # ensure indices are not out-of-bounds
    idx = plentiful_idx[plentiful_idx < len(xyz)]
    # compute translated coords centered at xyz_query
    xyz_os = xyz[idx] - xyz_query
    # encode samples by octant
    octants = (xyz_os[:,0] < 0) + 2*(xyz_os[:,1] < 0) + 4*(xyz_os[:,2] < 0)
    # allow m priority picks for each quadrant
    '''
    assert(knn_max >= 8)
    OR allow the function to return up to 8 samples if knn_max < 8
    '''
    m = max(knn_max // 8, 1)
    selected = np.zeros(len(idx), dtype=bool)
    count, target = 0, min(knn_max, len(idx))
    for octant in np.unique(octants):
        pick = np.where(octants == octant)[0][:m]
        selected[pick] = True
        count += len(pick)
    # complete the quota
    if count < target:
        candidates = np.where(selected == False)[0]
        selected[candidates[:target-count]] = True
    if debug:
        octant_codes = ','.join([str(c) for c in octants])
        return idx[selected], selected, octant_codes, np.r_[octants][selected]
    else:
        return np.where(selected)[0]

def find_neighbours2(xyz, xyz_queries, radius=None, knn_min=2, knn_max=9, tree=None, p_distance_exponent=2):
    """
    Find up to knn_max neighbouring points giving preference to selecting samples from each octant
    This implementation seeks to replicate `nearest_neighbor_search` in "gstatsim3d_util.py"

    Parameters
    ----------
        xyz : numpy.ndarray, shape=(nT,3)
            x,y,z coordinates of all training samples in (rotated+scaled) search space
        xyz_queries : numpy.ndarray, shape=(nQ,3)
            x,y,z coordinates of all queried locations
        radius : numpy.ndarray, shape=(3,) OR float
            search radius
        knn_min : int
            minimum number of unconditional neighbours
        knn_max : int
            maximum number of neighbouring samples to return
        tree : None or scipy.spatial cKDTree
            search object for making nearest neighbour query
        p_distance_exponent : int
            for example, 2 specifies L2 norm

    Returns
    -------
        d : numpy.ndarray, shape (nQ, knn_max)
            distances of neighbouring points, one row per query location
        idx_near : numpy.ndarray, shape (nQ, knn_max)
            index of the neighbouring points in xyz
        tree : scipy.spatial cKDTree
            kD-tree for re-use
    """
    num_samples = len(xyz)

    if num_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=int), tree

    num_neighbours = min(8 * knn_max, num_samples) #expanded to cater for octants

    if radius is not None:
        radius = np.array(radius).flatten()
        if len(radius) == 1:
            radius = np.full(xyz.shape[1], radius[0])
        xyz = xyz / radius[np.newaxis, :]
        xyz_queries = xyz_queries / radius[np.newaxis, :]

    if tree is None:
        tree = cKDTree(xyz)

    d, idx_near = tree.query(xyz_queries, num_neighbours, p=p_distance_exponent)
    if radius is not None:
        column_number = np.array(len(d) * [[c for c in range(num_neighbours)]])
        mask = (column_number > knn_min) & (d > 1.)
        idx_near[mask] = num_samples
    # The following would work if we impose no distance constraint,
    # then, the tree would always return the same number of neighbours
    '''
    selection = [octant_search(xyz, xyz_queries[r], idx_near[r], knn_max)
                 for r in range(len(xyz_queries))]
    selection = np.array(selection)
    d = np.take_along_axis(d, selection, axis=1)
    idx_near = np.take_along_axis(idx_near, selection, axis=1)
    '''
    width = d.shape[1]
    for r in range(len(xyz_queries)):
        selected = octant_search(xyz, xyz_queries[r], idx_near[r], knn_max)
        num_ignore = width - len(selected)
        d[r] = np.r_[d[r, selected], [np.inf] * num_ignore]
        idx_near[r] = np.r_[idx_near[r, selected], [num_samples] * num_ignore]

    return d, idx_near, tree


def get_neighbourhood_mean(y, idx_near):
    y_nan = np.concatenate((y, [np.nan]))
    return np.nanmean(y_nan[idx_near], axis=1)
