import numpy as np


def create_voxel_grid(minimum, maximum, num_voxels):
    """
    Creates the initial voxel grid.

    input: a numpy array containing the minimum coordinates for the voxel grid.
           a numpy array containing the maximum coordinates for the voxel grid.
           the number of voxels we want to use.
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


def carve(voxel_grid, silhouettes, proj_mats):
    """
    Carves out the object from the silhouettes.

    input: a numpy array representing the initial voxel grid.
           a numpy array containing all the silhouettes of the images.
           a numpy array containing all the corresponding projection matrices.
    output: a numpy array containing the coordinate points of the voxels that
              make up the reconstruction.
    """

    threshold = silhouettes.shape[0] * 0.8

    new_voxels = []

    for i in range(voxel_grid.shape[0]):

        # gets the homogenous img coordinates
        homogenous_voxel = np.append(voxel_grid[i], 1)
        img_coors = proj_mats @ homogenous_voxel

        # gets the cartesian image coordinates
        x = np.ndarray.round(img_coors[:, 0] / img_coors[:, 2]).astype(int)
        y = np.ndarray.round(img_coors[:, 1] / img_coors[:, 2]).astype(int)
        z = np.arange(len(x))

        #  removes out of bounds indices
        remove_x = np.where(x >= silhouettes[0].shape[1])
        remove_y = np.where(y >= silhouettes[0].shape[0])
        remove_indices = np.concatenate((remove_x, remove_y), axis=None)

        if not len(remove_indices) == 0:
            x = np.delete(x, remove_indices)
            y = np.delete(y, remove_indices)
            z = np.delete(z, remove_indices)

        values = silhouettes[z, y, x]

        # counts the number of values equal to 255 and if count is greater
        #   than threshold, then voxel is included
        p = np.count_nonzero(values == 255)

        if p >= threshold:
            new_voxels.append(voxel_grid[i])

    return np.array(new_voxels)
