import tifffile as tif
import h5py
import numpy as np
import os
import cv2


def read_image_from_ims_file(path_to_ims_file):
    """
    Reads image data together with relevant metadata from an Imaris ims file and returns the image data together
    with the extracted metadata.

    :param path_to_ims_file: path to the ims file from which the data should be extracted (string)
    :return: image_data_array (numpy.ndarray), metadata_dict (dict)
    """
    # print status message
    print(f'Read "{path_to_ims_file}" ...')
    # open ims file
    with h5py.File(path_to_ims_file, 'r') as f:
        # generate list of available channels
        channel_list = [ch_id for ch_id in f['DataSetInfo'] if ch_id.startswith('Channel')]
        # initialize empty lists for storing the data arrays and names of each channel
        image_data_array = []
        channel_names = []

        # iterate channels
        for channel in channel_list:
            # read data array
            image_data_array.append(np.array(f['DataSet']['ResolutionLevel 0']['TimePoint 0'][channel]['Data']))
            # read channel names
            channel_names.append(f['DataSetInfo'][channel].attrs['Name'].tobytes().decode('ascii', 'ignore'))

        # combine 3D arrays into a 4D array
        image_data_array = np.stack(image_data_array)

        # initialize empty dict for voxel and image sizes
        voxel_size = {}
        image_size = {}
        # read min and max metric coordinates as well as pixel size in all three dimensions and calculate voxel size
        for i, d in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            # read the highest metrical coordinate
            max_coord = float(f['DataSetInfo']['Image'].attrs[f'ExtMax{i}'].tobytes().decode('ascii', 'decode'))
            # read the lowest metrical coordinate
            min_coord = float(f['DataSetInfo']['Image'].attrs[f'ExtMin{i}'].tobytes().decode('ascii', 'decode'))
            # read the pixel size
            pixel_size = int(f['DataSetInfo']['Image'].attrs[d].tobytes().decode('ascii', 'decode'))

            # calculate metrical voxel size
            voxel_size[d] = (max_coord - min_coord) / pixel_size

            # add dimension size to image size dict
            image_size[d] = pixel_size

    # crop array size to image size
    image_data_array = np.transpose(image_data_array[:, :image_size['Z'], :image_size['Y'], :image_size['X']],
                                    (1, 0, 2, 3))

    # generate meta data dict
    metadata_dict = {'axes': 'ZCYX',
                     'axes_info': 'ZCYX',
                     'channels': len(channel_names),
                     'slices': image_size['Z'],
                     'hyperstack': True,
                     'mode': 'grayscale',
                     'channel_names': [s.lower() for s in channel_names],
                     'image_size': image_size,
                     'voxel_size': voxel_size,
                     'original_file': path_to_ims_file.split(sep='/')[-1]}

    # return image data array and dictionary with relevant metadata
    return image_data_array, metadata_dict


def read_ome_tiff_image_and_metadata(path_to_ome_tiff_file):
    """
    Opens an OME TIFF file and returns a numpy array of the image data together with a metadata containing dictionary
    :param path_to_ome_tiff_file: path to an OME TIFF file (string)
    :return: image data (numpy.ndarray), metadata (dict)
    """
    with tif.TiffFile(path_to_ome_tiff_file) as f:
        # read image data array
        image_data_array = f.asarray()
        # read metadata
        metadata = f.imagej_metadata
        # if channel_names are present in the metadata convert them to a list
        if 'channel_names' in metadata:
            metadata['channel_names'] = eval(metadata['channel_names'])

    # return the metadata
    return image_data_array, metadata


def save_ome_tiff_file(image_data_array,
                       metadata_dict,
                       path_to_new_ome_file,
                       default_prefix='image_',
                       default_suffix='ome.tif'):
    """
    Saves passed image and metadata as OME TIFF file. If a directory is passed es destination for saving the image,
    images are stored in consecutive order (dafault_prefix_{number}.default_suffix), if filename is passed image is
    stored utilizing this file name.

    :param image_data_array: array with the image pixel data (numpy.ndarray)
    :param metadata_dict: dictionary with all necessary metadata for the passed image (dict)
    :param path_to_new_ome_file: path to a directory or file name (string)
    :param default_prefix: (string) if not passed default value 'image' is used
    :param default_suffix: (string) if not passed default value 'ome.tif' is used
    """

    # check if axis is defined in metadata_dict
    if not 'axes' in metadata_dict:
        # add default axis order to metadata dict
        metadata_dict['axes'] = 'CZYX'

    # check if path for storing the image ends with the correct file extension (.ome.tif)
    if not path_to_new_ome_file.endswith(default_suffix):
        # if not, consider it as directory and make it if it does not exist
        if not os.path.exists(path_to_new_ome_file):
            os.makedirs(path_to_new_ome_file)
            # print status message
            print(f'Created new directory ({path_to_new_ome_file})!')

    # check if passed path is a directory
    if os.path.isdir(path_to_new_ome_file):
        # get a list of all files in the directory that match the prefix and suffix
        files = [f for f in os.listdir(path_to_new_ome_file) if
                 f.startswith(default_prefix) and f.endswith(default_suffix)]

        # Extract the index numbers from the filenames and sort them
        indices = sorted([int(f[len(default_prefix):-len(default_suffix)]) for f in files])

        # Check for the next available index
        if not indices:
            next_index = 1
        else:
            next_index = indices[-1] + 1
        # Construct the filename for the new file
        new_filename = f"{default_prefix}_{next_index}.{default_suffix}"
        path_to_new_ome_file = os.path.join(path_to_new_ome_file, new_filename)

    # save the image data array as ome-tiff file
    tif.imwrite(path_to_new_ome_file,
                image_data_array,
                shape=image_data_array.shape,
                imagej=True,
                metadata=metadata_dict)
    # print status message
    print(f'Saved file at {path_to_new_ome_file}!')


def save_simple_tif_file(image_array, path):
    # save the image data array as ome-tiff file
    tif.imwrite(path,
                image_array,
                shape=image_array.shape,
                imagej=True)


def save_slices_as_jpg(image_3d, folder_path, prefix="slice"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, slice_2d in enumerate(image_3d):
        filename = os.path.join(folder_path, f"{prefix}_{i}.jpg")
        cv2.imwrite(filename, slice_2d * 255)  # Assuming image_3d has floating point values between 0 and 1

