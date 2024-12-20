# Auxiliary functions to perform nearest neighbour search, compute covariance expressions and render 2D blocks
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
#--------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import time
from scipy import special
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def timeit(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        suffix = ""
        if 'desc' in kw:
            suffix = f"({kw['desc']})"
        res = func(*arg, **kw)
        t2 = time.time()
        print('%s%s took %.6fs' % (func.__name__, suffix, t2 - t1))
        return res
    return wrapper

#======================================================================

class Gridding:
    """
    Provide methods for making uniformly spaced, rectilinear 3D grids

    Note: These are not actually required for our experiments as the
          irregularly shaped inference locations are given by a block model.
    """
    def make_grid(xmin, xmax, ymin, ymax, zmin, zmax, res, predict=False):
        """
        Generate coordinates for output of gridded data  
        
        Parameters
        ----------
            xmin : float, int
                minimum x extent
            xmax : float, int
                maximum x extent
            ymin : float, int
                minimum y extent
            ymax : float, int
                maximum y extent
            zmin : float, int
                minimum z extent
            zmax : float, int
                maximum z extent
            res : float, int
                grid cell resolution
            prediction : bool
                if true, flip y-axis and compute intervals inclusive of max value
        
        Returns
        -------
            grid_xyz : numpy.ndarray
                x,y,z array of coordinates
            rows : int
                number of rows 
            cols : int 
                number of columns
            height : int
                number of vertical layers
        """
        if predict:
            cols = int(np.ceil((xmax - xmin)/res))
            rows = int(np.ceil((ymax - ymin)/res))
            height = int(np.ceil((zmax - zmin + res)/res))
            x = np.linspace(xmin, xmin+(cols*res), num=cols, endpoint=False)
            y = np.linspace(ymin, ymin+(rows*res), num=rows, endpoint=False)
            z = np.linspace(zmin, zmin+(height*res), num=height, endpoint=False)
        else:
            x = np.arange(xmin, xmax, res)
            y = np.arange(ymin, ymax, res)
            z = np.arange(zmin, zmax, res)
            cols = len(x)
            rows = len(y)
            height = len(z)
        xx, yy, zz = np.meshgrid(x,y,z)
        if predict:
            yy = np.flip(yy)
        #x = np.reshape(xx, (int(rows)*int(cols), 1))
        #y = np.reshape(yy, (int(rows)*int(cols), 1))
        #prediction_grid_xy = np.concatenate((x,y), axis = 1)
        grid_xyz = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

        return grid_xyz, cols, rows, height

    def prediction_grid(xmin, xmax, ymin, ymax, zmin, zmax, res):
        """
        Make a regular prediction grid given axes limits
        
        Returns
        -------
            prediction_grid_xyz : numpy.ndarray
                x,y,z array of coordinates
        """ 
        return Gridding.make_grid(xmin, xmax, ymin, ymax, zmin, zmax, res, predict=True)[0]

    def grid_data(df, xx, yy, zz, vv, res):
        """
        Grid conditioning data
        
        Parameters
        ----------
            df : pandas DataFrame 
                dataframe of conditioning data and coordinates
            xx : string 
                column name for x coordinates of input data frame
            yy : string
                column name for y coordinates of input data frame
            zz : string
                column name for z coordinates of input data frame
            vv : string
                column for values (or data variable) of input data frame
            res : float, int
                grid cell resolution
        
        Returns
        -------
            df_grid : pandas DataFrame
                dataframe of gridded data
            grid_matrix : numpy.ndarray
                matrix of gridded data
            rows : int
                number of rows in grid_matrix
            cols : int
                number of columns in grid_matrix
        """ 
        df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z", vv: "V"})

        xmin = df['X'].min()
        xmax = df['X'].max()
        ymin = df['Y'].min()
        ymax = df['Y'].max()
        zmin = df['Z'].min()
        zmax = df['Z'].max()

        # make array of grid coordinates
        grid_coord, cols, rows, height = Gridding.make_grid(xmin, xmax, ymin, ymax, zmin, zmax, res)

        df = df[['X','Y','Z','V']] 
        np_data = df.to_numpy() 
        np_resize = np.copy(np_data) 
        origin = np.array([xmin,ymin,zmin])
        resolution = np.array([res,res,res])
        
        # shift and re-scale the data by subtracting origin and dividing by resolution
        np_resize[:,:3] = np.rint((np_resize[:,:3]-origin)/resolution) 

        grid_sum = np.zeros((cols,rows,height))
        grid_count = np.copy(grid_sum) 

        for i in range(np_data.shape[0]):
            xindex = np.int32(np_resize[i,0])
            yindex = np.int32(np_resize[i,1])
            zindex = np.int32(np_resize[i,2])

            if ((xindex >= cols) | (yindex >= rows) | (zindex >= height)):
                continue

            grid_sum[xindex,yindex,zindex] += np_data[i,3]
            grid_count[xindex,yindex,zindex] += 1

        np.seterr(invalid='ignore') 
        grid_matrix = np.divide(grid_sum, grid_count) 
        grid_array = np.reshape(grid_matrix,[rows*cols*height]) 
        grid_sum = np.reshape(grid_sum,[rows*cols*height]) 
        grid_count = np.reshape(grid_count,[rows*cols*height]) 

        # make dataframe    
        grid_total = np.array([grid_coord[:,0], grid_coord[:,1], grid_coord[:,2],
                               grid_sum, grid_count, grid_array])    
        df_grid = pd.DataFrame(grid_total.T,
                               columns = ['X', 'Y', 'Z', 'Sum', 'Count', 'V'])
        grid_matrix = np.flipud(grid_matrix)

        return df_grid, grid_matrix, rows, cols, height

#======================================================================

class NearestNeighbor:
    """
    Implement octant-based nearest neighbor search and auxiliary functions
    that compute the centroid, L2-norm distances and enumerate samples by octant.
    """
    def center(arrayx, arrayy, arrayz, centerx, centery, centerz):
        """
        Shift data points so that grid cell of interest is at the origin
        
        Parameters
        ----------
            arrayx : numpy.ndarray
                x coordinates of data
            arrayy : numpy.ndarray
                y coordinates of data
            arrayz : numpy.ndarray
                z coordinates of data
            centerx : float
                x coordinate of grid cell of interest
            centery : float
                y coordinate of grid cell of interest
            centerz : float
                z coordinate of grid cell of interest

        Returns
        -------
            centered_array : numpy.ndarray [[..x..],[..y..],[..z..]]
                array of coordinates that are shifted with respect to grid cell of interest
        """
        return np.array([arrayx - centerx, arrayy - centery, arrayz - centerz])

    def distance_calculator(centered_array):
        """
        Compute distances between coordinates and the origin
        
        Parameters
        ----------
            centered_array : numpy.ndarray
                array of coordinates
        
        Returns
        -------
            dist : numpy.ndarray
                array of distances between coordinates and origin
        """
        return np.linalg.norm(centered_array, axis=0)

    def octant_calculator(centered_array):
        """
        Enumerate samples by octant
        
        Parameters
        ----------
            centered_array : numpy.ndarray  [[..x..],[..y..],[..z..]]
                array of coordinates relative to queried location (or grid cell coordinates)
        
        Returns
        -------
            angles : numpy.ndarray
                encoding of position relative to x, y, z axes, using a 3 bit representation
                (bz,by,bx) where b* is set to 1 when test value is negative, 0 otherwise.
        """
        return (centered_array[0] < 0) + 2 * (centered_array[1] < 0) + 4 * (centered_array[2] < 0)

    def nearest_neighbor_search(radius, num_points, loc, data2, min_points=2):
        """
        Nearest neighbor octant search (modified for 3D data) including
        replenishment in cases where eligible samples in an octant are exhausted.
        
        Parameters
        ----------
            radius : int, float
                search radius
            num_points : int
                number of points to search for
            loc : numpy.ndarray
                coordinates for grid cell of interest
            data2 : pandas DataFrame
                with column names ['X','Y','Z'] for coordinates and 'V' for values
        
        Returns
        -------
            near : numpy.ndarray
                nearest neighbors with columns corresponding to x, y, z, v.
        """ 
        
        locx = loc[0]
        locy = loc[1]
        locz = loc[2]
        data = data2.copy()
        centered_array = NearestNeighbor.center(data['X'].values, data['Y'].values,
                                                data['Z'].values, locx, locy, locz)
        data["dist"] = NearestNeighbor.distance_calculator(centered_array)
        retained = np.flatnonzero(data.dist < radius) #use contiguous indices

        #relax radius constraint if fewer than min_points were found
        if len(retained) < min_points:
            preference = np.argsort(data.dist)
            retained = preference[:min_points]
        
        data = data.iloc[retained]
        data = data.sort_values('dist', ascending = True)
        #data['relX'] = centered_array[0,retained]
        #data['relY'] = centered_array[1,retained]
        #data['relZ'] = centered_array[2,retained]
        data["octant"] = NearestNeighbor.octant_calculator(centered_array[:,retained])
        data["used"] = False
        octant_quota = max(num_points // 8, 1)
        required_cols = ['X','Y','Z','V']
        neighbors = []

        for i in range(8): #strive for per-octant proportional representation
            candidates = data[data.octant == i].index
            octant = data.loc[candidates[:octant_quota],required_cols].values
            data.loc[candidates[:octant_quota], "used"] = True
            for row in octant:
                neighbors.append(row)

        target_number = min(num_points, len(data))
        if len(neighbors) < target_number: #not enough samples from some octants
            data = data[data["used"] == False]
            i = 0
            while len(neighbors) < target_number:
                neighbors.append(data.iloc[i][required_cols].values)
                i += 1

        return np.array(neighbors, dtype=float)

#======================================================================

def make_rotation_matrix():
    """
    Use rtc_trend_alignment.compute_any_rotation_and_scaling_matrix instead
    """
    raise NotImplementedError

def domain_id_to_column_name(domain_id):
    lz = domain_id // 1000
    gz = (domain_id % 1000) // 100
    por = (domain_id % 100) // 10
    rock_type = domain_id % 10
    domain_key = f"LZ{lz}_{gz}_{por}_{rock_type}"
    return domain_key

def clean_ellipsoid_file_extract_dataframe(domain_key, filepath=None):
    """
    Process messy ellipsoid data file, returns a pandas.Dataframe with
    columns ["variable", domain_key] thus each rows contains key, value.
    """
    erase_start_sentinel = lambda s : s.replace('%VAR ', '')

    if filepath is None:
        filepath = "data/future bench variograms.txt"
    df_rs = pd.read_csv(filepath, skiprows=1, comment='#', sep=r'\s*[,=]\s*',
                        engine='python', converters={'%VAR domain': erase_start_sentinel})
    df_rs = df_rs.rename(columns = {'%VAR domain': "variable"})
    df_rs = df_rs.loc[:, ["variable", domain_key]] #trimmed data for relevant domain
    return df_rs

def get_angles_ellipse_radii(df, domain_key):
    deg2rad = np.pi / 180.
    get_ = lambda df,k,v: float(df[df['variable']==v][k].values[0])
    params = np.r_[get_(df, domain_key, "angle1"), #intepret as azimuth
                   get_(df, domain_key, "angle2"), #intepret as plunge
                   get_(df, domain_key, "angle3"), #intepret as dip
                   get_(df, domain_key, "search1x"),
                   get_(df, domain_key, "search1y"),
                   get_(df, domain_key, "search1z")]
    print('{} params: {}'.format(domain_key, dict(zip(['azi','plg','dip','eX','eY','eZ'], params))))
    params[:3] *= deg2rad #convert angles to radian
    return params

def make_scatter_2d(x, y, v, min_v=None, max_v=None, symbol='.', symbsiz=30,
                    subplotargs=None, palette='Jet', xlabel='X', ylabel='Y',
                    graphtitle='', cbtitle='', cbticklabels=None, cbfontsz=None,
                    min_xy=None, max_xy=None, sharex=False, sharey=False,
                    savefile=None, interactive=True):
    if subplotargs is None:
        fig = plt.figure(figsize=(8,6.5))
    else:
        rows, cols, num = subplotargs
        plt.gcf().add_subplot(rows, cols, num)
    ax = plt.gca()
    im = ax.scatter(x, y, c=v, vmin=min_v, vmax=max_v, marker=symbol, s=symbsiz, cmap=palette)
    plt.title(graphtitle, fontsize=10)
    if sharex:
        plt.tick_params(
            axis='x',     # changes apply to the x-axis
            which='both', # both major and minor ticks are affected
            bottom=False, # ticks along the bottom edge are off
            top=False,    # ticks along the top edge are off
            labelbottom=False)
    else:
        plt.xlabel(xlabel)
    if sharey:
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False)
    else:
        plt.ylabel(ylabel)
    plt.locator_params(nbins=5)
    if min_xy is None and max_xy is None:
        plt.axis('scaled')
    else:
        ax.set_xlim([min_xy[0], max_xy[0]])
        ax.set_ylim([min_xy[1], max_xy[1]])
    # make colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='6.5%', pad=0.1)
    cbar_ticks = np.linspace(min_v, max_v, 11) if cbticklabels is None else cbticklabels[0]
    cbar = plt.colorbar(im, ticks=cbar_ticks, cax=cax)
    cbar.set_label(cbtitle, rotation=270, labelpad=10)
    if cbticklabels is not None:
        cbfs = cbfontsz if cbfontsz is not None else 10
        cbar.ax.set_yticklabels(cbticklabels[1], fontsize=cbfs)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0.05)
        #https://dev.to/siddhantkcode/optimizing-matplotlib-performance-handling-memory-leaks-efficiently-5cj2
        plt.clf()
        plt.close()
    if subplotargs is None and interactive:
        plt.show()
    return ax

#======================================================================

class Covariance:
    """
    Implement covariance calculations associated with Kriging

    Note: Offer additional support for Matern(nu) covariance function
    beside the exponential, spherical and Gaussian covariance functions.
    """
    def covar(effective_lag, sill, nugget, nu, vtype):
        """
        Compute covariance
        
        Parameters
        ----------
            effective_lag : np.array or float
                lag distance that is normalized to a range of 1
            sill : int, float
                sill of variogram
            nugget : int, float
                nugget of variogram
            nu : float
                Matern kernel smoothness parameter (only applicable when vtype=='Matern')
            vtype : string
                type of variogram model (Exponential, Gaussian, Matern or Spherical)
        Note
        ----
        Added support for Matern covariance function based on skgstat.models

        Raises
        ------
        AtrributeError : if vtype is not 'Exponential', 'Gaussian', 'Matern' or 'Spherical'

        Returns
        -------
            c : numpy.ndarray
                covariance
        """

        if vtype.lower() == 'exponential':
            c = (sill - nugget) * np.exp(-3 * effective_lag)
        elif vtype.lower() == 'gaussian':
            c = (sill - nugget) * np.exp(-3 * np.square(effective_lag))
        elif vtype.lower() == 'spherical':
            c = sill - nugget - 1.5 * effective_lag + 0.5 * np.power(effective_lag, 3)
            c[effective_lag > 1] = sill - 1
        elif vtype.lower() == 'matern': #the following formula has been verified
            #for a = range/2, use expression in skgstat.models
            if isinstance(effective_lag, np.ndarray):
                c = np.ones(effective_lag.shape)
                mask = effective_lag > 0
                c[mask] = (sill - nugget) * ((2 / special.gamma(nu)) *
                    np.power((2 * effective_lag[mask] * np.sqrt(nu)), nu) *
                    special.kv(nu, 4 * effective_lag[mask] * np.sqrt(nu)))
            elif isinstance(effective_lag, float):
                c = 1 if effective_lag == 0 else \
                    (sill - nugget) * ((2 / special.gamma(nu)) *
                    np.power((2 * effective_lag * np.sqrt(nu)), nu) *
                    special.kv(nu, 4 * effective_lag * np.sqrt(nu)))
        else: 
            raise AttributeError("vtype must be 'Exponential', 'Gaussian', 'Matern' or 'Spherical'")
        return c

    def make_covariance_matrix(coords, vario, origin=None):
        """
        Make covariance matrix showing covariances between each pair of input coordinates
        
        Parameters
        ----------
            coords : numpy.ndarray
                coordinates of n data points
            vario : list
                list of variogram parameters [nugget, range, sill, shape, vtype, R, S]
                nugget, range, sill are common to all vtypes, supported vtypes
                include {'Exponential', 'Spherical', 'Matern', 'Gaussian'}
                shape (nu) describes the smoothness of the Matern kernel, where
                0.5 corresponds to exponential, lim_{shape->inf} corresponds to
                the Gaussian kernel, in practice, this is often upper-bounded by 20.
                R is the rotation matrix, either numpy.array of shape (3,3) or None
                S is the scaling array, either numpy.array of shape (3,) or None
        
        Returns
        -------
            covariance_matrix : numpy.ndarray 
                nxn matrix of covariance between n points
        """
        nugget, effective_range, sill, nu, vtype, R, S = vario
        if R is None:
            R = np.eye(3, dtype=float)
        if S is None: #assume ellipsoid is isotropic
            S = effective_range * np.ones(3)
        #otherwise, S=[sX,sY,sZ] describes the actual range w.r.t. the X, Y and Z axes
        transformation = np.diag(1./S) @ R
        if origin is None:
            origin = np.mean(coords, axis=0)
        mat = np.matmul(coords - origin, transformation.T)
        #spatial transformation is always required because the scaling part
        #produces range-normalised coordinates (lags) required by .covar
        effective_lag = pairwise_distances(mat, mat) #shape:(n,n)
        covariance_matrix = Covariance.covar(effective_lag, sill, nugget, nu, vtype)

        return covariance_matrix

    def make_covariance_array(coords1, coord2, vario, origin=None):
        """
        Make covariance array showing covariances between the data points and query point
        
        Parameters
        ----------
            coords1 : numpy.ndarray of shape (n,3)
                coordinates of n data points
            coord2 : numpy.ndarray of shape (3,)
                coordinates of query point of interest (i.e. grid cell being simulated)
            vario : list
                list of variogram parameters [nugget, range, sill, shape, vtype, R, S]
                nugget, range, sill are common to all vtypes, supported vtypes
                include {'Exponential', 'Spherical', 'Matern', 'Gaussian'}
                shape (nu) describes the smoothness of the Matern kernel
                R is the rotation matrix, either numpy.array of shape (3,3) or None
                S is the scaling array, either numpy.array of shape (3,) or None
        
        Returns
        -------
            covariance_array : numpy.ndarray
                nx1 array of covariance between n points and grid cell of interest
        """
        nugget, effective_range, sill, nu, vtype, R, S = vario
        if R is None:
            R = np.eye(3, dtype=float)
        if S is None:
            S = effective_range * np.ones(3)
        transformation = np.diag(1./S) @ R
        if origin is None:
            origin = np.mean(coords1, axis=0)
        #spatial transformation is always required because the scaling part
        #produces range-normalised coordinates (lags) required by .covar
        mat1 = np.matmul(coords1 - origin, transformation.T)
        mat2 = np.matmul(coord2 - origin, transformation.T)
        #np.tile(mat2, len(mat1)) is unnecessary as numpy broadcasting
        #rule ensures the size of objects in matrix ops are compatible
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1)) #shape:(n,)
        covariance_array = Covariance.covar(effective_lag, sill, nugget, nu, vtype)

        return covariance_array

#======================================================================

__all__ = ['Gridding', 'NearestNeighbor', 'Covariance']

def __dir__():
    return __all__

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f'module {__name__} has no attribute {name}')
    return globals()[name]
