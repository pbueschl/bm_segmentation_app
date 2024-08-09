import os
import tifffile as tif
import numpy as np
import nibabel as nib
import sys
import shutil
import gc

sys.path.append('..')
from utils import ims_files, utils, image_io, image_spliter
import subprocess

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def get_full_dataset_name(dataset_id, path_to_nnunet_results):
    """

    :param dataset_id:
    :param input_folder:
    :param output_folder:
    :param path_to_nnunet_results:
    :return:
    """
    # look for available datasets
    list_of_datasets = os.listdir(path_to_nnunet_results)
    # iterate datasets
    for dataset_name in list_of_datasets:
        # print(dataset_name, f'Dataset{int(dataset_id):03}')
        # break iterating if dataset name belongs to given id
        if dataset_name.startswith(f'Dataset{int(dataset_id):03}'):
            print(dataset_name, dataset_id)
            return dataset_name
        else:
            print('No matching Dataset found! List of available datasets: \n', list_of_datasets)


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
    # if len(channel_names) > 1:
    # update axes in metadata_dict
    metadata_dict['axes'] = metadata_dict['axes_info'] = 'ZCYX'
    # reset hyperstack flag in metadata_dict
    metadata_dict['hyperstack'] = True
    # else:
    # update axes in metadata_dict
    #    metadata_dict['axes'] = metadata_dict['axes_info'] = 'ZYX'
    # reset hyperstack flag in metadata_dict
    #    metadata_dict['hyperstack'] = False

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


def check_ram_usage(all_variables):
    # Iterate over the whole list where dir( )
    # is stored.
    for name in all_variables:
        try:
            # Print the item if it doesn't start with '__'
            if not name.startswith('__'):
                myvalue = eval(name)
                size = sys.getsizeof(myvalue) / 1024 ** 3
                if type(myvalue) is np.ndarray:
                    datatype = myvalue.dtype
                else:
                    datatype = None
                print(name, "is", type(myvalue), datatype, f"Size: {size:.3f} GB")

        except:
            print(f"EXCEPTION for variable: {name} ")


def inference(args, channels_of_interest):
    """

    :param args:
    :param channels_of_interest:
    :return:
    """

    # check if output directory exists, if not, create it
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # create cache directory for saving files needed for running inference
    path_to_cache = 'cache'

    if os.path.exists(path_to_cache):
        shutil.rmtree(path_to_cache)
    os.makedirs(path_to_cache)
    path_to_input_cache = 'cache/input'
    path_to_output_cache = 'cache/output'
    if not os.path.exists(path_to_input_cache):
        os.makedirs(path_to_input_cache)
    if not os.path.exists(path_to_output_cache):
        os.makedirs(path_to_output_cache)

    # instantiate image split processor
    max_pixels = 67 * 2 * 4000 * 3000
    processor = image_spliter.IMSProcessor(args.input,
                                           path_to_cache,
                                           max_pixels,
                                           channels_of_interest)
    # split image into chunks
    processor.split_image()

    # print status message
    print(f'Run mask generation...')

    # get nnunet environment variables
    path_to_nnunet_raw, path_to_nnunet_preprocessed, path_to_nnunet_results = utils.get_nnunet_environment_variables()
    # get full dataset name
    full_dataset_name = get_full_dataset_name(args.dataset_id, path_to_nnunet_results)

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(path_to_nnunet_results, full_dataset_name, f'nnUNetTrainer__nnUNetPlans__{args.nnunet_config}'),
        use_folds=args.nnunet_folds,
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(path_to_input_cache),
                                 join(path_to_output_cache),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    gc.collect()
    # print status message
    print(f'Load generated mask...')
    # remove unnecessary variables from memory
    # stitch segmentation mask and load it
    data_array = processor.stitch_segmentation_mask()
    # print status message
    print(f'Save resulting OME TIFF files at "{args.output}"...')

    # ------------------------ save predicted mask ------------------------
    # define mask name
    mask_name = f'predicted_{args.mask_selection}_mask'
    # define path to output file
    path_to_output_ome_tiff_file = os.path.join(args.output,
                                                f'{os.path.splitext(os.path.basename(args.input))[0]}__{args.mask_selection}_mask.ome.tif')
    # save resulting ome tiff file
    create_metadict_and_save_ome_tiff_file(data_array[:,np.newaxis], [mask_name], processor.input_metadata, path_to_output_ome_tiff_file)

    shutil.rmtree(path_to_cache)
    # remove deleted variable from memory
    gc.collect()
    # print status message
    print(f'Save results as OME TIFF files at "{args.output}"...')

    # print status message
    print(f'Finished mask generation for {args.input}!')
