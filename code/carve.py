from os import remove
from matplotlib.pyplot import xcorr
import numpy as np


def create_voxel_grid(minimum, maximum, num_voxels):
    """
    Creates the initial voxel grid.

    input: a numpy array containing the minimum coordinates for the voxel grid, 
           a numpy array containing the maximum coordinates for the voxel grid, 
           and the number of voxels we want to use.
    output: a numpy array containing the coordinates for the voxels where each 
            row is a x,y,z coordiante for a single voxel and the distance 
            between each voxel.
    """

    x_length = maximum[0] - minimum[0]
    y_length = maximum[1] - minimum[1]
    z_length = maximum[2] - minimum[2]

    grid_volume = x_length * y_length * z_length
    voxel_volume = grid_volume / num_voxels
    voxel_side_length = voxel_volume ** (1. / 3)
    half_side_length = voxel_side_length / 2

    num_x = x_length // voxel_side_length
    num_y = y_length // voxel_side_length
    num_z = z_length // voxel_side_length

    voxel_points = []

    # creates the voxel grid
    for x in range(int(num_x)):
        for y in range(int(num_y)):
            for z in range(int(num_z)):
                x_coor = minimum[0] + \
                    (x * voxel_side_length + half_side_length)
                y_coor = minimum[1] + \
                    (y * voxel_side_length + half_side_length)
                z_coor = minimum[2] + \
                    (z * voxel_side_length + half_side_length)

                voxel_points.append([x_coor, y_coor, z_coor])

    return np.array(voxel_points), voxel_side_length


def carve(voxel_grid, silhouette, proj_mat):

    # adds 1 to voxel grid coordinates to make them homogenous coordiantes
    ones = np.ones((voxel_grid.shape[0], 1))
    homogenous_voxel = np.append(voxel_grid, ones, axis=1)

    # coverts homogenous coordiantes to image coordiantes
    img_coors = np.transpose(
        np.matmul(proj_mat, np.transpose(homogenous_voxel)))

    # turns homogenous coordinates into image coordinates
    x_coors = np.ndarray.round(img_coors[:, 0] / img_coors[:, 2]).astype(int)
    y_coors = np.ndarray.round(img_coors[:, 1] / img_coors[:, 2]).astype(int)

    # removes out of bounds indices
    remove_x = np.where(x_coors >= silhouette.shape[1])
    remove_y = np.where(y_coors >= silhouette.shape[0])
    remove_indices = np.concatenate((remove_x, remove_y), axis=None)

    removed_voxels = voxel_grid[remove_indices]

    if not len(remove_indices) == 0:
        x_coors = np.delete(x_coors, remove_indices)
        y_coors = np.delete(y_coors, remove_indices)
        voxel_grid = np.delete(voxel_grid, remove_indices)

    # gets the voxels that are in the silhouette
    silhouette_vals = silhouette[y_coors, x_coors]
    voxel_indices = np.where(silhouette_vals == 255)
    new_voxels = voxel_grid[voxel_indices]
    return np.vstack([new_voxels, removed_voxels])
