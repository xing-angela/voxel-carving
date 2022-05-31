import cv2
import numpy as np
import open3d as o3d


def get_silhouettes(images):
    """
    Returns an array of the silhouettes of the images.

    input: an array of images.
    output: a numpy array containing the silhouettes of the images.
    """

    all_silhouettes = []
    for i in range(len(images)):
        if images[i] is not None:
            # convert the image to grayscale
            grayscale = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            blur = cv2.blur(grayscale, (3, 3))

            # get the silhouette of the image
            _, silhouette = cv2.threshold(
                blur, 40, 255, cv2.THRESH_BINARY)

            all_silhouettes.append(silhouette)

    return np.array(all_silhouettes)


def get_camera_info(path, image_files):
    """
    Returns an array containing the projection matrices for the cameras
    of each image and the corrdinate points of the cameras. 

    input: a string representing the path to the camera calibration info
           a numpy array containing the image file names.
    output: a numpy array containing all the camera projection matrices.
            a numpy array containing the camera position information.
    """

    image_files = np.array(image_files)

    all_proj_mats = np.zeros((len(image_files), 3, 4))
    with open(path) as file:
        for i, line in enumerate(file.readlines()):
            if i > 0:
                info = line.split()
                mat_index = np.where(image_files == info[0])[0][0]

                # gets the projection matrix
                k = np.array([[float(info[1]), float(info[2]), float(info[3])],
                              [float(info[4]), float(info[5]), float(info[6])],
                              [float(info[7]), float(info[8]), float(info[9])]])

                rt = np.array([[float(info[10]), float(info[11]),
                                float(info[12]), float(info[19])],
                               [float(info[13]), float(info[14]),
                                float(info[15]), float(info[20])],
                               [float(info[16]), float(info[17]),
                                float(info[18]), float(info[21])]])

                curr_proj_mat = k @ rt
                all_proj_mats[mat_index] = curr_proj_mat

    return all_proj_mats


def show_voxel_model(points, voxel_size):
    """
    Creates a voxel grid based on the 3d points and colors found.

    input: a numpy array that contains the coordinate points of the voxels
           the side length of each voxel
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # voxels are a random color
    # pcd.colors = o3d.utility.Vector3dVector(
    #     np.random.uniform(0, 1, size=(points.shape[0], 3)))

    # voxels are black
    pcd.colors = o3d.utility.Vector3dVector(
        np.zeros((points.shape[0], points.shape[1])))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])
