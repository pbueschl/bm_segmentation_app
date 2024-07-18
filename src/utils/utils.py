import os

import numpy as np
import nibabel as nib
import tifffile as tif
from utils.nnUNet_nifti_files import write_nnUNet_training_nifti_files
from utils import image_io
import gc

channels_of_interest = {
    'dapi': ['dapi'],
    'endomucin': ['endomucin'],
    'endoglin': ['endoglin', 'endoglin_bad'],
    'cxcl12': ['cxcl12'],
    'collagen': ['collagen'],
    'foxp3': ['foxp3']
}

labels_of_interest = {
    'dapi': ['pred_label_channel_1', 'predmpick_label_channel_1', 'dapi mask'],
    'sinusoids': ['pred_label_channel_2', 'predmpick_label_channel_2', 'cnnseg sinusoids', 'sinusoids mask'],
    'transitional_vessels': ['cnnseg tv'],
    'arteries': ['predmpick_label_channel_4', 'cnnseg arteries']
}


def select_channels(path_to_input_tif_file, path_to_coi_output, channels_of_interest_dict):
    """
    Returns a subset of the passed channels of the data array. The selection is based on the passed metadata and
    channels of interest
    :param data_array: source image data (numpy.ndarray)
    :param metadata: metadata of the source file (dict)
    :param channels_of_interest_dict: channels that should be returned in form of a dict holding lists of possible
                                      channel names (value) for a given unified channel name (key) (dict)
    :return:data array containing the requested channels (numpy.ndarray)
    """
    # load input data array
    data_array, metadata = image_io.read_ome_tiff_image_and_metadata(path_to_input_tif_file)
    # get channel_list from metadata
    available_channels_list = metadata['channel_names']
    # if available channels list is not of type list, convert it to type list
    if type(available_channels_list) is not list:
        available_channels_list = eval(available_channels_list)
    # print status message
    print(f"The following channels are available in the input file: {available_channels_list}")
    # initialize empty list for indices of channels of interest
    channels_of_interest_indices_list = []
    # iterate channels of interest
    for unified_channel_name, possible_channel_names_list in channels_of_interest_dict.items():
        # iterate list of possible channel names
        for channel_name in possible_channel_names_list:
            # check if possible channel name is present in passed metadata
            if any(channel_name == ch for ch in available_channels_list):
                # get index of channel
                channels_of_interest_indices_list.append(available_channels_list.index(channel_name))
                # stop for loop
                break
            # if for loop was not stopped raise an error, that channel of interest is not present in the passed data.
            raise ValueError(f'The expected channel "{unified_channel_name}" does not seem to exist in the passed data'
                             f'file. Please check the data and/or the channel naming convention of the source file.')
    # update metadata
    metadata['channel_names'] = list(channels_of_interest_dict.keys())
    metadata['channels'] = len(metadata['channel_names'])
    # print status message
    print(f"Return the following channels: {metadata['channel_names']}")

    data_array = data_array[:, channels_of_interest_indices_list, :, :]
    # save resulting ome tiff file
    tif.imwrite(path_to_coi_output,
                data_array,
                shape=data_array.shape,
                imagej=True,
                metadata=metadata)

    del data_array
    gc.collect()


def unify_channels(image_data_array, metadata_dict, channels_dict=channels_of_interest,
                   labels_dict=labels_of_interest):
    """
    This function unifies the channels of the passed image data array and returns the reduced image data array together
    with the updated metadata dictionary. For reducing the channels it looks the passed dictionaries of channels and
    labels. The keys of these two dictionaries give the name of the desired channel or label while the corresponding
    value lists names that might be used in the original file for naming this type of channel. In short, this function
    unifies the available data channels as well as the names of the channels in the metadata dict
    :param image_data_array: 4D image data array of structure (Z,C,Y,X)
    :param metadata_dict: dictionary that holds the metadata (in particular the channel names) (dict)
    :param channels_dict: dictionary that holds the desired channels of the returned image data.
                          {'unified_channel_name': [list of possible channel names used at the passed file(s)], ... }
    :param labels_dict: dictionary that holds the desired labels of the returned image data.
                        {'unified_label_name': [list of possible channel names used at the passed file(s)], ... }
    :return: image_data_dict (Z,C,Y,X) (numpy.ndarray), metadata_dict
    """
    # read list of channel_names from metadata
    metadata_channel_names = metadata_dict['channel_names']
    # convert channel names to list if it is not already of type list
    if type(metadata_channel_names) is not list:
        metadata_channel_names = eval(metadata_channel_names)

    # initialize empty dict for name and index of available channels
    available_channels_dict = {}
    # iterate passed channels of interest
    for global_channel_name, possible_channel_names_list in channels_dict.items():
        # iterate through list of possible channel names
        for channel_name in possible_channel_names_list:
            # check if channel is present in passed metadata
            if any(channel_name == ch for ch in metadata_channel_names):
                # add global channel name and position in metadata list to the available channels dict
                available_channels_dict[global_channel_name] = metadata_channel_names.index(channel_name)
                # stop for loop
                break

    # check if arteries label is desired even if collagen channel is not available at the passed image
    if 'arteries' in labels_dict and set(channels_dict['collagen']).isdisjoint(metadata_channel_names):
        # remove arteries label from labels dict
        del labels_dict['arteries']
    # initialize empty dict for name and index of available labels
    available_labels_dict = {}
    # iterate passed labels of interest
    for global_label_name, possible_label_names_list in labels_dict.items():
        # iterate through list of possible label names
        for label_name in possible_label_names_list:
            # check if label is present in passed metadata
            if any(label_name == ch for ch in metadata_channel_names):
                # add global label name and position in metadata list to the available channels dict
                available_labels_dict[global_label_name] = metadata_channel_names.index(label_name)
                # stop for loop
                break

    # read channel indices from available channels dict
    channel_indices = list(available_channels_dict.values())
    # read channel names from available channels dict
    channel_names = ['channel_' + s for s in available_channels_dict.keys()]

    # read channel indices from available labels dict
    channel_indices.extend(list(available_labels_dict.values()))
    # read channel names from available channels dict
    channel_names.extend(['label_' + s for s in available_labels_dict.keys()])

    # reduce image channels by masking with channel indices
    image_data_array = image_data_array[:, channel_indices, :, :]

    # update metadata dictionary channel names and number of channels
    metadata_dict['channel_names'] = channel_names
    metadata_dict['channels'] = len(channel_names)

    # return metadata dict and image data array
    return image_data_array, metadata_dict

def write_nifti_file(output_nifti_file_path, data_array, resolution):
    """
    Saves data array together with the resolution as an affine matrix in a nifti file
    :param output_nifti_file_path: path to the desired output file (string)
    :param data_array: 4D data
    :param resolution:
    :return:
    """
    # transpose data array to have XYZ order and convert it to uint8
    data_array = data_array.transpose((2, 1, 0)).astype(np.uint8)
    # Create a NIfTI image header with the appropriate dimensions and spacing information
    affine = np.array([[resolution[2], 0, 0, 0], [0, resolution[1], 0, 0], [0, 0, resolution[0], 0], [0, 0, 0, 1]])
    # create nifti image object
    nii_image = nib.Nifti1Image(data_array, affine=affine)
    # save the nifti file
    nib.save(nii_image, output_nifti_file_path)


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


def save_data_channels_for_inference(image_data_array, metadata_dict, output_path, image_id=0):
    # read voxel size from metadata
    voxel_size = metadata_dict['voxel_size']
    # if voxel_size is not of type dict, convert it to type dict
    if type(voxel_size) is not dict:
        voxel_size = eval(voxel_size)
    # convert voxel size from dict to list of order (Z,Y,X)
    resolution = [voxel_size['Z'], voxel_size['Y'], voxel_size['X']]

    # iterate channels
    for i in range(image_data_array.shape[1]):
        # save channels in nifti files
        write_nnUNet_training_nifti_files(output_path, image_data_array[:, i, :, :], resolution, image_id, i, 'infimg')


def get_nnunet_environment_variables():
    # open bashrc file
    with open(os.path.join(os.environ['HOME'], '.bashrc')) as f:
        # read lines where nnUNet environment variables are exported
        lines = [line for line in f.readlines() if line.startswith('export nnUNet_')]
    # extract environement paths from environment variable definitions
    for s in lines:
        if 'nnUNet_raw' in s:
            raw = s.split(sep='"')[-2]
        if 'nnUNet_preprocessed' in s:
            preprocessed = s.split(sep='"')[-2]
        if 'nnUNet_results' in s:
            results = s.split(sep='"')[-2]

    # return environment variables
    return raw, preprocessed, results