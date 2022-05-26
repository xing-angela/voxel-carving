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
