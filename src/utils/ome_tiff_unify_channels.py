
"""
Definition of dictionaries for channels and labels of interest. The dictionary key gives the new unified channel name
while the values are lists of possible channel names that are used in the input files. The unify_channels function
iterates through these lists and looks for the first matching channel name. This means that the order maters and the
fist possible channel name in the list has the highest priority as just one channel is picked for the unifed data file.
"""
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

import tifffile as tif
import argparse
import os


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


def main():
    """
    Main function for unifying the data channels and names of a given OME TIFF file or for all OME TIFF files in a
    given directory.
    """
    # Create the parser object
    parser = argparse.ArgumentParser(description='Unifies data channels and names in either a single OME TIFF file or '
                                                 'several OME TIFF files of a given directory.')

    # Add arguments for the input path and output directory
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input OME TIFF file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Directory for storing the unified OME TIFF image file')

    # Parse the arguments
    args = parser.parse_args()

    # check if output directory exists, if not create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # check if passed input path belongs to a file or directory
    if os.path.isfile(args.input):
        # put file name as single element in list of file names
        ome_tiff_files = [args.input]
    else:
        # get a list of all files in the directory
        file_list = os.listdir(args.input)

        # filter the list to only include .ome.tif files
        ome_tiff_files = [os.path.join(args.input, f) for f in file_list if f.endswith('.ome.tif')]

    # iterate ome tiff files
    for i, f in enumerate(ome_tiff_files, start=1):
        # read image data array and metadata from the passed file
        image_data_array, metadata_dict = read_ome_tiff_image_and_metadata(f)
        # call function for unifying image data
        image_data_array, metadata_dict = unify_channels(image_data_array, metadata_dict,
                                                         channels_of_interest, labels_of_interest)
        # define output file name
        path_to_unified_ome_tiff_file = os.path.join(args.output,
                                                     f'{os.path.basename(f)[:-8]}_unified.ome.tif')

        # save the image data array as OME TIFF file
        tif.imwrite(path_to_unified_ome_tiff_file,
                    image_data_array,
                    shape=image_data_array.shape,
                    imagej=True,
                    metadata=metadata_dict)

        # print status message
        print(f'[{i}/{len(ome_tiff_files)}] Saved unified image data at {path_to_unified_ome_tiff_file}!')


if __name__ == "__main__":
    main()
