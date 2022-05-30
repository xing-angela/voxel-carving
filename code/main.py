import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from process import get_silhouettes, get_camera_info, get_bounding_box, \
    show_voxel_model
from carve import create_voxel_grid, carve


def parse_args():
    """
    Perform command-line argument parsing (gets the data).
    """

    parser = argparse.ArgumentParser(
        description="Get voxel carving images")
    parser.add_argument(
        '--images',
        required=True,
        choices=['dino', 'dino_test', 'temple', 'templeSR'],
        help='Which image sequence to use')

    return parser.parse_args()


def main():
    args = parse_args()

    # gets the image files
    data_dir = os.path.join('../data/', args.images)
    image_files = os.listdir(data_dir)

    images = []
    for image_file in image_files:
        images.append(cv2.imread(os.path.join(data_dir, image_file)))
    print(f'{len(image_files)} images for {args.images} sequence loaded')

    # gets the silhouettes of the images
    silhouettes = get_silhouettes(images)
    print(f'got silhouettes for {args.images} sequence')

    # gets the projection matrices of the cameras for each image
    proj_mats, cam_coors = get_camera_info('../data/' + args.images +
                                           '_info/' + args.images + '_par.txt')
    print(f'got projection matrices and camera coordinates ' +
          f'for cameras of {args.images} images')

    # gets the bounding box information
    # min_coors, max_coors = get_bounding_box(cam_coors)
    # print(min_coors)
    # print(max_coors)

    # dino
    # min_coors = np.array([-0.041897, 0.001126, -0.037845])
    # max_coors = np.array([0.030897, 0.088227, 0.035495])

    # temple
    # min_coors = np.array([-0.054568, 0.001728, -0.042945])
    # max_coors = np.array([0.047855, 0.161892, 0.032236])

    # templeSR
    min_coors = np.array([-0.073568, 0.021728, -0.012445])
    max_coors = np.array([0.028855, 0.181892, 0.062736])

    # creates the initial voxel grid
    num_voxels = 100000
    print(f'showing initial voxel grid with {num_voxels} voxels')
    voxel_grid, voxel_length = create_voxel_grid(
        min_coors, max_coors, num_voxels)
    show_voxel_model(voxel_grid, voxel_length)

    curr_voxel_grid = voxel_grid

    print('showing the carved voxel grid')
    # plt.imshow(silhouettes[2], cmap='gray')
    # plt.show()
    # curr_voxel_grid = carve(curr_voxel_grid, silhouettes[2], proj_mats[2])
    # show_voxel_model(curr_voxel_grid, voxel_length)
    for i in range(len(silhouettes)):
        # plt.imshow(silhouettes[i], cmap='gray')
        # plt.show()
        curr_voxel_grid = carve(curr_voxel_grid, silhouettes[i], proj_mats[i])

    show_voxel_model(curr_voxel_grid, voxel_length)


if __name__ == '__main__':
    main()
