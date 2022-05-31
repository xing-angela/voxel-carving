import argparse
import os
import numpy as np
import cv2
from process import get_silhouettes, get_camera_info, show_voxel_model
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
        choices=['dino', 'temple'],
        help='Which image sequence to use')
    parser.add_argument(
        '--num_voxels',
        type=int,
        default=200000,
        help='The number of voxels to use')

    return parser.parse_args()


def main():
    args = parse_args()

    # gets the image files
    data_dir = os.path.join('../data/', args.images)
    image_files = np.array(os.listdir(data_dir))

    images = []
    for image_file in image_files:
        images.append(cv2.imread(os.path.join(data_dir, image_file)))
    print(f'{len(image_files)} images for {args.images} sequence loaded')

    # gets the silhouettes of the images
    silhouettes = get_silhouettes(images)
    print(f'got silhouettes for {args.images} sequence')

    # gets the projection matrices of the cameras for each image
    proj_mats = get_camera_info('../data/' + args.images +
                                '_info/' + args.images + '_par.txt', image_files)
    print(f'got projection matrices and camera coordinates ' +
          f'for cameras of {args.images} images')

    # sets the bounding box information based on the dataset
    min_coors = np.zeros(3)
    max_coors = np.zeros(3)
    if (args.images == 'dino'):
        min_coors = np.array([-0.041897, 0.001126, -0.037845])
        max_coors = np.array([0.030897, 0.088227, 0.035495])
    if (args.images == 'temple'):
        min_coors = np.array([-0.054568, 0.001728, -0.042945])
        max_coors = np.array([0.047855, 0.161892, 0.032236])

    # creates the initial voxel grid
    num_voxels = args.num_voxels
    print(f'showing initial voxel grid with {num_voxels} voxels')
    voxel_grid, voxel_length = create_voxel_grid(
        min_coors, max_coors, num_voxels)
    show_voxel_model(voxel_grid, voxel_length)

    # creates the reconstruction
    print('showing the carved voxel grid')
    new_voxels = carve(voxel_grid, silhouettes, proj_mats)
    show_voxel_model(new_voxels, voxel_length)


if __name__ == '__main__':
    main()
