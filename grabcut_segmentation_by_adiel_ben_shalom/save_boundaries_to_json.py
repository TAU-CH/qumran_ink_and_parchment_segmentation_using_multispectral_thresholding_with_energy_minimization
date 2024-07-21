import os
from tracing_utils_moore import trace_boundary
import cv2
from matplotlib.colors import Colormap
from remove_jp_paper import remove_jp_paper
from cut_fragment import cut_fragment
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import numpy as np
from skimage.draw import polygon_perimeter
import json


def save_boundaries_to_json(image_file, output_file):
    """save boundary of fragment to json file

    Parameters
    ----------
    image_file : str
        The full path to fragment location (with image name)
    output_file: str
        The full path to the output Json file name
    
    Returns
    -------
        The method writes the the boundary of the fragment to json file
        The json file contains both the fragment's external polygon boundary as 
        well as internal halls boundary
        exterior borders are counter clockwise and interior borders are clockwise
        The is according to GetJSON format see here
        https://datatracker.ietf.org/doc/html/rfc7946#page-9
        and here
        https://mariadb.com/kb/en/mariadb/st_geomfromgeojson/
    """
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image_bw1 = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image_bw = (image_bw1 > 0).astype(np.uint8)

   # _, axes = plt.subplots(1, 3, sharex=True, sharey=True)
   # axes[0].imshow(image)
   # axes[1].imshow(image_bw1)
    #axes[2].imshow(image_bw)

    borders = trace_boundary(image_bw)
    output = {"type": "Polygon"}
    coordinates = []
    colors = []
    output_image = np.zeros_like(image)
    debug = False
    remove_duplicates = True
    for index, border in enumerate(borders):
        rr_bord, cc_bord = border
        rr, cc = polygon_perimeter(rr_bord, cc_bord)
        c_r_indices = np.stack((rr, cc))
        c_r_indices = np.flip(c_r_indices, axis=1)

        if remove_duplicates:
            unique_elements = np.abs(np.diff(c_r_indices, axis=1)).sum(axis=0) > 0 
            unique_elements = np.concatenate([[True], unique_elements])
            c_r_indices = c_r_indices[:, unique_elements]
        coordinates.append(c_r_indices.T.tolist())

        color = np.array(cmap.gist_rainbow(index / len(borders))) * 255
        colors.append(color)
        output_image[c_r_indices[0], c_r_indices[1], :] = color

        if debug:
            if index == 2:
                for poly_index in range(len(c_r_indices[0])):
                    plt.plot(c_r_indices[0][poly_index], c_r_indices[1][poly_index], 'r+')
                    plt.text(c_r_indices[0][poly_index], c_r_indices[1][poly_index], poly_index)
                plt.axis('equal')
                plt.show()

    # _, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(image)
    # axes[1].imshow(output_image)
    # plt.show()
    output_dir=os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output['coordinates'] = coordinates
    with open(output_file, 'w') as fh:
        json.dump(output, fh)

    print('save_boundaries_to_json done')



def display_json(json_file, image_file):
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_file, 'r') as fh:
        all_coordinates = json.load(fh)
    
    plt.imshow(image)
    for coordinate in all_coordinates['coordinates']:
        coordinate_np = np.stack(coordinate)
        plt.scatter(coordinate_np[:, 1], coordinate_np[:, 0])
    plt.show()
