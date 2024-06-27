from matplotlib.font_manager import findSystemFonts
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple, Union
from numpy.random import randint, choice, rand
from scipy.spatial.transform import Rotation
from scipy.io import loadmat
from scipy.ndimage import rotate
from scipy.interpolate import griddata
import pathlib

__all__ = ["random_volume", "colin27_volume", "digimouse_volume", "usc195_volume"]


def random_char_in_2d_volume(img_size: Tuple[int, int], min_ratio_of_ones: int = 0.01) -> np.ndarray:
    """
    Generates a random ASCII character inside an image and returns it as binary
    :param img_size: size of the image in shape of [rows, columns]
    :param min_ratio_of_ones: Minimum ratio of ones over total pixels in the output mask
    :return: the logical image with the ASCII character randomly placed inside
    """
    mask = Image.new('L', (1, 1), 0)
    min_number_of_ones = min_ratio_of_ones * img_size[0] * img_size[1]
    while np.array(mask).sum() < min_number_of_ones:
        mask.close()
        mask = Image.new('L', img_size, 0)
        mask_draw = ImageDraw.Draw(mask)
        x, y = randint(img_size[0] * 0.9), randint(img_size[1] * 0.9)
        rand_char = chr(randint(33, 126))
        font = ImageFont.truetype(choice(findSystemFonts()), size=randint(30, 100))
        mask_draw.text((x, y), rand_char, fill=1, font=font)
        # mask = mask.rotate(rand() * 360)
    return np.array(np.ceil(mask), dtype=np.uint8)


def random_char_in_3d_volume(volume_size):
    """
    Places a random ASCII character with a random thickness randomly inside a 3D volume. The character is also randomly
    rotated around each axis.
    :param volume_size: size of the volume in shape of [rows, columns, height]
    :return:the logical image with the ASCII character randomly placed inside
    """
    # Generate a random 2D character
    char_2d = random_char_in_2d_volume(volume_size[0: 2])
    # Dimension expansion for easier stacking of the 2D image
    char_2d = np.expand_dims(char_2d, 2)
    # Randomly select a z-start and z-end to give volume to the letter
    z_start = randint(volume_size[2])
    z_end = randint(z_start, volume_size[2])
    binary_volume = np.zeros(volume_size, dtype=np.float32)
    # Stack the same character multiple times over the z axis
    binary_volume[:, :, z_start: z_end] = char_2d
    # Random rotation around the x and y-axis
    # binary_volume = rotate(binary_volume, rand() * 360, (1, 2), reshape=False, order=0)
    # binary_volume = rotate(binary_volume, rand() * 360, (0, 2), reshape=False, order=0)
    # Some interpolation is done in the rotation, we need to make them part of the object
    return np.ceil(binary_volume).astype(np.uint8)


def random_polygon_in_2d_volume(num_sides: int, ctr_to_vtx_range: Union[int, Tuple[int, int]],
                                image_size: Tuple[int, int]) -> np.ndarray:
    """
    Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
    Generates a random polygon with the specified number of sides with the value of 1 inside a zeros 2D matrix.
    If the length of ctr_to_vtx_range is 2, center to vertex distance for each vertex is randomly generated
    between the range provided. If only a single number is provided, the vertices will have the same distance.
    The shape is randomly rotated and placed in the volume.
    :param num_sides: Number of sides of the polygon
    :param ctr_to_vtx_range: In form of [min_distance, max_distance], specifies the distance range from a vertex to the
    centroid
    :param image_size: Size of the image, in shape of [rows, columns]
    :return: A binary image with the size of volume_size, with 1 indicating the polygon and 0 indicating the absence of
    the polygon
    """
    rows, columns = image_size
    # Create a polygon around the origin
    vertices = np.arange(num_sides)
    angles = (vertices * 2 * np.pi) / num_sides

    if isinstance(ctr_to_vtx_range, int):
        ctr_to_vtx_dists = np.ones(num_sides, dtype=np.int) * ctr_to_vtx_range
    else:
        ctr_to_vtx_dists = np.random.randint(*ctr_to_vtx_range, num_sides)
    # Element-wise multiplication of each angle and vertex-centroid distance
    xyz = np.zeros((num_sides, 3), dtype=np.float32)
    xyz[:, 0] = ctr_to_vtx_dists * np.cos(angles)
    xyz[:, 1] = ctr_to_vtx_dists * np.sin(angles)
    # Rotate the coordinates by a random angle between 0 and 2pi
    angle_to_rotate = 360 * np.random.rand()
    xyz = Rotation.from_euler('Z', angle_to_rotate, degrees=True).apply(xyz)
    # Get a random center location between max_distance and (columns - max_distance).
    # This will ensure it's always in the image.
    max_distance = max(ctr_to_vtx_range) if not isinstance(ctr_to_vtx_range, int) else ctr_to_vtx_range

    x_center = max_distance + (columns - 2 * max_distance) * np.random.rand()
    y_center = max_distance + (rows - 2 * max_distance) * np.random.rand()
    # Translate the image so that the center is at (x_center, y_center) rather than at (0,0).
    xyz[:, 0] += x_center
    xyz[:, 1] += y_center
    # Create a binary mask out of the coordinates
    xy = xyz[:, :2]
    # PIL only supports a list of floating point tuples for coordinates
    xy = [(pt[0], pt[1]) for pt in xy]
    mask = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def random_polygon_in_3d_volume(num_sides: int, ctr_to_vtx_range: Union[int, Tuple[int, int]],
                                volume_size: Tuple[int, int, int]):
    """
    Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
    Create a random polygon around the origin with given number of sides and ctr_to_vtx_range and randomly
    rotates and places it in a 3D volume specified by volume_size.
    3D point placement logic used from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    :param num_sides: Number of sides for the polygon. Must be greater than 4
    :param ctr_to_vtx_range: In form of [min_distance, max_distance], specifies the distance range from a vertex to
    the centroid
    :param volume_size: Size of the output volume, in the form of [rows, columns, sheets]
    :return: A binary volume with the size of volume_size, with 1 indicating the polygon and 0 indicating the absence
    of the polygon
    """
    v = np.arange(num_sides)
    # Randomly generate distances for each vertex if a range is specified
    if isinstance(ctr_to_vtx_range, int):
        ctr_to_vtx_dists = np.ones(num_sides, dtype=np.int) * ctr_to_vtx_range
    else:
        ctr_to_vtx_dists = np.random.randint(*ctr_to_vtx_range, num_sides)
    z = (v / num_sides) * 2 * ctr_to_vtx_dists - ctr_to_vtx_dists
    phi = (v / num_sides) * 2 * np.pi
    xyz = np.zeros((num_sides, 3), dtype=np.float32)
    xyz[:, 0] = np.sqrt(np.power(ctr_to_vtx_dists, 2) - np.power(z, 2)) * np.cos(phi)
    xyz[:, 1] = np.sqrt(np.power(ctr_to_vtx_dists, 2) - np.power(z, 2)) * np.sin(phi)
    xyz[:, 2] = z
    # Randomly rotate the vertices around each axis in 3D
    rand_rot = Rotation.from_euler('xyz', (rand() * 360, rand() * 360, rand() * 360), degrees=True)
    xyz = np.floor(rand_rot.apply(xyz))
    # Translate the image so that the center is at (x_center, y_center, z_center) rather than at (0,0,0)
    max_distance = max(ctr_to_vtx_range) if not isinstance(ctr_to_vtx_range, int) else ctr_to_vtx_range

    x_center = max_distance if max_distance == volume_size[0] - max_distance else\
        randint(*sorted((max_distance, volume_size[0] - max_distance)))
    y_center = max_distance if max_distance == volume_size[1] - max_distance else\
        randint(*sorted((max_distance, volume_size[1] - max_distance)))
    z_center = max_distance if max_distance == volume_size[2] - max_distance else\
        randint(*sorted((max_distance, volume_size[2] - max_distance)))
    xyz[:, 0] += x_center
    xyz[:, 1] += y_center
    xyz[:, 2] += z_center
    grid_x, grid_y, grid_z = np.meshgrid(range(volume_size[0]), range(volume_size[1]), range(volume_size[2]))
    mask = griddata(xyz, np.ones(len(xyz)), (grid_x, grid_y, grid_z), fill_value=0, method="linear")
    return np.round(mask).astype(np.uint8)


def random_volume(volume_size: Union[Tuple[int, int], Tuple[int, int, int]], num_shapes: int) -> np.ndarray:
    """
    Generates a random volume with the specified number of shapes. Each shape can be either a character or a
    polygon. Each shape will have a unique label. If two shapes or more interest, that intersection will get its
    unique label as well.
    :param volume_size: size of the volume
    :param num_shapes: number of shapes to be placed inside the volume
    :return: rasterized volume with each shape uniquely labeled, and each intersection uniquely labeled
    """
    assert len(volume_size) == 2 or len(volume_size) == 3
    volume = np.zeros(volume_size, dtype=np.uint8)
    # If number of props is zero, return an empty volume
    if num_shapes != 0:
        if len(volume_size) == 2:
            random_char_generator = random_char_in_2d_volume
            random_polygon_generator = random_polygon_in_2d_volume
            min_polygon_sides = 3
        else:
            random_char_generator = random_char_in_3d_volume
            random_polygon_generator = random_polygon_in_3d_volume
            min_polygon_sides = 4
        last_prop_idx = 1
        # Make sure the vertex-centroid range is selected from the smallest axis
        min_dim = min(volume_size)
        ctr_vtx_range = (np.round(min_dim / 5), np.round(min_dim / 2))
        for i in range(num_shapes):
            # Randomly choose between placing a character or a polygon in the volume
            if rand() > 0.5:
                curr_prop = random_char_generator(volume_size)
            else:
                num_sides = randint(min_polygon_sides, 8)
                curr_prop = random_polygon_generator(num_sides, ctr_vtx_range, volume_size)
            curr_prop = last_prop_idx * curr_prop
            volume += curr_prop
            # Finds the last unused label for the next iteration
            while last_prop_idx in volume:
                last_prop_idx += 1
    if len(volume.shape) == 2:
        volume = np.expand_dims(volume, 0)
    # Normalize the labels (make them go from 0 to N)
    unique_indicies = np.unique(volume)
    for i, idx in enumerate(unique_indicies):
        volume[volume == idx] = i
    return volume


def colin27_volume() -> np.ndarray:
    """
    :return: Colin 27 Atlas as a 3D numpy array
    """
    vol_dir = str(pathlib.Path(__file__).parent.resolve().joinpath("colin27_v3.mat"))
    return loadmat(vol_dir)["colin27"]


def digimouse_volume() -> np.ndarray:
    """
    :return: Digimouse Atlas as a 3D numpy array
    """
    vol_dir = str(pathlib.Path(__file__).parent.resolve().joinpath("digimouse.mat"))
    return loadmat(vol_dir)["digimouse"]


def usc195_volume() -> np.ndarray:
    """
    :return: USC-195 Atlas as a 3D numpy array
    """
    vol_dir = str(pathlib.Path(__file__).parent.resolve().joinpath("fullhead_atlas.mat"))
    return loadmat(vol_dir)["USC_atlas"]
