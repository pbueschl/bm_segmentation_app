import os
import tifffile as tif
import numpy as np
import nibabel as nib
import sys
import shutil

sys.path.append('..')
from utils import ims_files, utils
import subprocess


def get_inference_command(dataset_id, input_folder, output_folder):
    """

    :param dataset_id:
    :param input_folder:
    :param output_folder:
    :return:
    """
    # get nnUNet results folder
    path_to_nnunet_results = os.environ['nnUNet_results']
    # look for available datasets
    list_of_datasets = os.listdir(path_to_nnunet_results)
    # iterate datasets
    for dataset_name in list_of_datasets:
        print(dataset_name, f'Dataset{int(dataset_id):03}')
        # break iterating if dataset name belongs to given id
        if dataset_name.startswith(f'Dataset{int(dataset_id):03}'):
            print(dataset_name, dataset_id)
            break
        else:
            print('No matching Dataset found! List of available datasets: \n', list_of_datasets)

    # combine dataset path to inference instruction txt file
    path_to_inference_instruction_file = os.path.join(path_to_nnunet_results,
                                                      dataset_name,
                                                      'inference_instructions.txt')
    # read inference instruction txt file
    with open(path_to_inference_instruction_file, 'r') as f:
        file_contend = f.read()

    # split contend
    file_contend_split = file_contend.split(sep='\n')
    # filter file contend for nnUNet commands
    nnunet_commands = [cm for cm in file_contend_split if cm.startswith('nnUNetv2')]

    # add input folder to inference command
    inference_command = nnunet_commands[0].replace('INPUT_FOLDER', input_folder)
    # add output folder to inference command
    inference_command = inference_command.replace('OUTPUT_FOLDER', output_folder)

    # define path for postprocessing output folder
    output_folder_postprocessing = os.path.join(output_folder, 'postproessing')
    # check if folder exists
    if not os.path.exists(output_folder_postprocessing):
        os.makedirs(output_folder_postprocessing)

    # add input folder to inference command
    postprocessing_command = nnunet_commands[1].replace('OUTPUT_FOLDER_PP', output_folder_postprocessing)
    # add output folder to inference command
    postprocessing_command = postprocessing_command.replace('OUTPUT_FOLDER', output_folder)

    # return prediction and postprocessing command
    return inference_command, postprocessing_command


def create_metadict_and_save_ome_tiff_file(data_array, channel_names, metadata_dict, path_to_output_file):
    """

    :param data_array:
    :param channel_names:
    :param metadata_dict:
    :param path_to_output_file:
    :return:
    """
    # convert channel names to list if it is not already of type list
    if type(channel_names) is not list:
        channel_names = eval(channel_names)
    # check dimension of data_array
    if data_array.shape[1] > 3:
        # update axes in metadata_dict
        metadata_dict['axes'] = metadata_dict['axes_info'] = 'ZCYX'
        # reset hyperstack flag in metadata_dict
        metadata_dict['hyperstack'] = True
    else:
        # update axes in metadata_dict
        metadata_dict['axes'] = metadata_dict['axes_info'] = 'ZYX'
        # reset hyperstack flag in metadata_dict
        metadata_dict['hyperstack'] = False

    # update channels information and names
    metadata_dict['channels'] = data_array.shape[1]
    metadata_dict['channel_names'] = [s.lower() for s in channel_names]

    # print status message
    print(f'Save concatenated data array as OME TIFF file at "{path_to_output_file}"...')
    # save resulting ome tiff file
    tif.imwrite(path_to_output_file,
                data_array,
                shape=data_array.shape,
                imagej=True,
                metadata=metadata_dict)


def inference(args, channels_of_interest):
    """

    :param args:
    :param channels_of_interest:
    :return:
    """

    # check if output directory exists, if not, create it
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # load ims file
    input_data_array, metadata = ims_files.read_image_from_ims_file(args.input)

    # print status message
    print(f'Select necessary channels...')
    # select needed channels
    inference_data_array = utils.select_channels(input_data_array, metadata, channels_of_interest)

    # create cache directory for saving files needed for running inference
    path_to_input_cache = 'cache/input'
    path_to_output_cache = 'cache/output'
    if not os.path.exists(path_to_input_cache):
        os.makedirs(path_to_input_cache)
    if not os.path.exists(path_to_output_cache):
        os.makedirs(path_to_output_cache)

    # print status message
    print(f'Save selected channels in cache...')

    # safe channels in a nifti files
    utils.save_data_channels_for_inference(inference_data_array, metadata, path_to_input_cache)

    # print status message
    print(f'Run mask generation...')

    # get nnUNet results folder
    path_to_nnunet_raw = os.environ['nnUNet_raw']
    # get nnUNet results folder
    path_to_nnunet_preprocessed = os.environ['nnUNet_preprocessed']
    # get nnUNet results folder
    path_to_nnunet_results = os.environ['nnUNet_results']
    # get home folder
    path_to_home_folder = os.environ['HOME']
    inference_command, postprocessing_command = get_inference_command(args.dataset_id, path_to_input_cache,
                                                                      path_to_output_cache)
    # construct command to run nnUNet_predict script
    bash_command = f' export nnUNet_raw="{path_to_nnunet_raw}";\
                           export nnUNet_preprocessed="{path_to_nnunet_preprocessed}";\
                           export nnUNet_results="{path_to_nnunet_results}";\
                            . {path_to_home_folder}/mambaforge-pypy3/bin/activate; \
                            mamba activate segtool; \
                            {inference_command}' #; \
    #                        {postprocessing_command}'
    # run inference
    subprocess.run(bash_command, shell=True, check=True, executable='/bin/bash')

    # print status message
    print(f'Load generated mask...')
    # load generated mask
    img = nib.load('cache/output/infimg_000.nii.gz')
    predicted_mask_array = img.get_fdata()

    # reshape predicted mask array and change boolen values to higher ones for better presentability
    predicted_mask_array = predicted_mask_array.transpose((2, 1, 0)) * 128
    # expand dimension of predicted mask array from ZYX to ZCYX
    predicted_mask_array = np.expand_dims(predicted_mask_array, axis=1).astype(np.uint8)
    # delete cache directory and files:
    shutil.rmtree('cache')

    # print status message
    print(f'Save resulting OME TIFF files at "{args.output}"...')

    # ------------------------ save predicted mask ------------------------
    # define mask name
    mask_name = f'predicted_{args.mask_selection}_mask'
    # define path to output file
    path_to_output_ome_tiff_file = os.path.join(args.output,
                                                f'{os.path.splitext(os.path.basename(args.input))[0]}__vessel_mask.ome.tif')
    # save resulting ome tiff file
    create_metadict_and_save_ome_tiff_file(predicted_mask_array, [mask_name], metadata, path_to_output_ome_tiff_file)

    # ------------------------ save predicted mask together with used signal channels ------------------------
    # define channel names
    channel_names = list(channels_of_interest.keys())
    channel_names.append(mask_name)

    # concatenate image data with the generated mask
    output_data_array = np.concatenate((inference_data_array, predicted_mask_array), axis=1)

    # define path to output file
    path_to_output_ome_tiff_file = os.path.join(args.output,
                                                f'{os.path.splitext(os.path.basename(args.input))[0]}__{"__".join(channel_names)}.ome.tif')
    # save resulting ome tiff file
    create_metadict_and_save_ome_tiff_file(output_data_array, channel_names, metadata, path_to_output_ome_tiff_file)

    # ------------------------ save predicted mask together with all channels from input file ------------------------
    # read list of channel names from metadata
    channel_names = metadata['channel_names']
    # convert channel names to list if it is not already of type list
    if type(channel_names) is not list:
        channel_names = eval(channel_names)
    # define channel names
    channel_names.append(mask_name)

    # concatenate image data with the generated mask
    output_data_array = np.concatenate((input_data_array, predicted_mask_array), axis=1)

    # define path to output file
    path_to_output_ome_tiff_file = os.path.join(args.output,
                                                f'{os.path.splitext(os.path.basename(args.input))[0]}__all_channels.ome.tif')
    # save resulting ome tiff file
    create_metadict_and_save_ome_tiff_file(output_data_array, channel_names, metadata, path_to_output_ome_tiff_file)

    # print status message
    print(f'Save concatenated data array as OME TIFF file at "{path_to_output_ome_tiff_file}"...')
    # save resulting ome tiff file
    create_metadict_and_save_ome_tiff_file()

    # print status message
    print(f'Finished mask generation for {args.input}!')
