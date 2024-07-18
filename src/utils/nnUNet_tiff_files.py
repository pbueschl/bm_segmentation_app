import argparse
import os
import tifffile as tif
import json
from utils import read_ome_tiff_image_and_metadata, channels_of_interest


def set_up_nnUNet_file_structure(path_to_nnUNet_dataset, dataset_id):
    # combine path to nnUNet dataset
    path_to_nnUNet_dataset = os.path.join(path_to_nnUNet_dataset, dataset_id)
    # check if output directory exists, if not create it
    if not os.path.exists(path_to_nnUNet_dataset):
        os.makedirs(path_to_nnUNet_dataset)

    # define path to nnUNet typic imagesTr directory
    path_to_images_tr = os.path.join(path_to_nnUNet_dataset, 'imagesTr')
    # check if output subdirectory imagesTr exists
    if not os.path.exists(path_to_images_tr):
        os.makedirs(path_to_images_tr)
    # define path to nnUNet typic labelsTr directory
    path_to_labels_tr = os.path.join(path_to_nnUNet_dataset, 'labelsTr')
    # check if output subdirectory labelsTr exists
    if not os.path.exists(path_to_labels_tr):
        os.makedirs(path_to_labels_tr)
    # return
    return path_to_nnUNet_dataset, path_to_images_tr, path_to_labels_tr


def write_nnUNet_training_tif_files(path_output_directory, data_array, resolution, image_nr, channel_nr, case_id):
    # define tiff file name
    path_to_tiff_file = os.path.join(path_output_directory, f'{case_id}_{image_nr:03d}_{channel_nr:04d}.tif')
    # define path to json file
    path_to_json_file = path_to_tiff_file[:-4] + '.json'
    # save data array in tiff file
    tif.imwrite(path_to_tiff_file, data_array)
    # create dict for spacing
    spacing = {'spacing': (resolution[0], resolution[1], resolution[2])}
    # create json file with channel resolution values
    with open(path_to_json_file, 'w') as f:
        json.dump(spacing, f)


def write_nnUNet_label_tif_files(path_output_directory, data_array, resolution, image_nr, case_id):
    # define tiff file name
    path_to_tiff_file = os.path.join(path_output_directory, f'{case_id}_{image_nr:03d}.tif')
    # define path to json file
    path_to_json_file = path_to_tiff_file[:-4] + '.json'
    # save data array in tiff file
    tif.imwrite(path_to_tiff_file, data_array)
    # create dict for spacing
    spacing = {'spacing': (resolution[0], resolution[1], resolution[2])}
    # create json file with channel resolution values
    with open(path_to_json_file, 'w') as f:
        json.dump(spacing, f)


def ome_tiff_to_nnUNet(path_to_ome_tiff_input_files,
                       path_to_nnUNet_dataset,
                       dataset_id,
                       dataset_abbreviation,
                       global_channel_ids,
                       global_label_id):
    # set up the folder structure for nnUNet
    path_to_nnUNet_dataset, path_to_images_tr, path_to_labels_tr = set_up_nnUNet_file_structure(path_to_nnUNet_dataset,
                                                                                                dataset_id)
    # get list of OME TIFF files
    ome_tiff_files = os.listdir(path_to_ome_tiff_input_files)
    # filter the list to only include .ome.tif files
    ome_tiff_files = [os.path.join(path_to_ome_tiff_input_files, f) for f in ome_tiff_files if f.endswith('.ome.tif')]

    # create dataset json file
    dataset_dict = {
        "channel_names": {str(i): k for i, k in enumerate(global_channel_ids.keys())},
        "labels": {
            "background": 0,
            global_label_id: 1,
        },
        "numTraining": int(len(ome_tiff_files)),
        "file_ending": ".tif"
    }

    # update global channels dictionary with a channel_id number and a prefixed channel name
    for i, k in enumerate(global_channel_ids.keys()):
        # add id number and prefixed name to global channels dictionary
        global_channel_ids[k] = {'name': f'channel_{k}',
                                 'id_nr': i}

    # iterate ome tiff files
    for i, f in enumerate(ome_tiff_files):
        # read data and meta data from ome tiff file
        image_data_array, metadata_dict = read_ome_tiff_image_and_metadata(f)
        # read channel names from metadata
        channel_names = metadata_dict['channel_names']
        # read voxel size from metadata
        voxel_size = eval(metadata_dict['voxel_size'])
        # convert voxel size from dict to list of order (Z,Y,X)
        resolution = [voxel_size['Z'], voxel_size['Y'], voxel_size['X']]

        # check if label channel is available
        if global_label_id in channel_names:
            # get channel number of label channel in current image data array
            ch_nr_local = channel_names.index(global_label_id)
            # read label data
            label_data_array = image_data_array[:, ch_nr_local, :, :]
            # convert labeled data array to boolen data array
            label_data_array = label_data_array.astype(bool)
            # save channel in nnUNet consistent structure
            write_nnUNet_label_tif_files(path_to_labels_tr, label_data_array,
                                         resolution, i, dataset_abbreviation)

            # iterate global channel ids
            for v in global_channel_ids.values():
                # check if channel is present in the current file
                if v['name'] in channel_names:
                    # get channel number at current image
                    ch_nr_local = channel_names.index(v['name'])
                    # read channel data
                    channel_data_array = image_data_array[:, ch_nr_local, :, :]
                    # save channel in nnUNet consistent structure
                    write_nnUNet_training_tif_files(path_to_images_tr, channel_data_array,
                                                    resolution, i, v['id_nr'], dataset_abbreviation)

            # print status message
            print(f'[{i + 1}/{len(ome_tiff_files)}] Converted "{f}" into nnUNet file structure!')

    # get number of trainign files
    num_training = int(len(os.listdir(path_to_labels_tr)) / 2)
    # update number of training date in dataset dict
    dataset_dict['numTraining'] = num_training
    # write dataset dict to JSON file
    with open(os.path.join(path_to_nnUNet_dataset, 'dataset.json'), "w") as f:
        json.dump(dataset_dict, f, indent=4)


def main():
    """

    """
    # create the parser object
    parser = argparse.ArgumentParser(description='Unifies data channels and names in either a single OME TIFF file or '
                                                 'several OME TIFF files of a given directory.')

    # add arguments for the input path and output directory
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input or directory of OME TIFF files')
    parser.add_argument('-o', '--output', required=True,
                        help='Directory for storing the nnUNet structured Dataset')
    parser.add_argument('-d', '--dataset_id', required=True,
                        help='Dataset name, should be of structure "DatasetXXX_NAME" where XXX refers to a 3 digit '
                             'id-number and NAME can be freely chosen.')
    parser.add_argument('-a', '--dataset_abbreviation', required=True,
                        help='short identifier as prefix to the single file names')
    parser.add_argument('-l', '--label', required=True,
                        help='string that identifies label channel in source data')
    # parse the arguments
    args = parser.parse_args()

    # call function for converting data from OME TIFF into nnUNet specific file structure
    ome_tiff_to_nnUNet(args.input,
                       args.output,
                       args.dataset_id,
                       args.dataset_abbreviation,
                       channels_of_interest,
                       args.label)

    # print status message
    print(f'Finished nnUNet conversion of dataset: {args.dataset_id}!')


if __name__ == "__main__":
    main()
