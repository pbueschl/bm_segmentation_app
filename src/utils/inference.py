import os
import tifffile as tif
import numpy as np
import nibabel as nib
import sys
import shutil
import gc
import time
from utils.utils import remove_old_cache_dirs
sys.path.append('..')
from utils import ims_files, utils, image_io, image_spliter
import subprocess
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.ensembling.ensemble import ensemble_folders
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder


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


def inference(args, channels_of_interest, gpu_id=0, patch_scaling_factor=None, debug_flag=False):
    """

    :param gpu_id:
    :param args:
    :param channels_of_interest:
    :return:
    """

    # check if output directory exists, if not, create it
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # remove cache directories that are older than 2 days
    remove_old_cache_dirs(os.getcwd(), '2d')
    # create cache directory for saving files needed for running inference
    path_to_cache = f'cache_{time.strftime("%Y%m%d_%H%M%S")}'
    # check if debug flag is set
    if debug_flag:
        # extend cach path by debug parent directory
        path_to_cache = os.path.join(os.getcwd(), 'debug', path_to_cache)
    # check if cache directory already exists, if so, remove it
    if os.path.exists(path_to_cache):
        shutil.rmtree(path_to_cache)
    os.makedirs(path_to_cache)
    path_to_input_cache = os.path.join(path_to_cache,'input')
    path_to_output_cache = os.path.join(path_to_cache,'output')
    if not os.path.exists(path_to_input_cache):
        os.makedirs(path_to_input_cache)
    if not os.path.exists(path_to_output_cache):
        os.makedirs(path_to_output_cache)

    # instantiate image split processor
    max_pixels = 67 * 2 * 3000 * 4000
    if patch_scaling_factor:
        max_pixels = int(patch_scaling_factor * max_pixels)

    processor = image_spliter.IMSProcessor(args.input,
                                           path_to_cache,
                                           max_pixels,
                                           channels_of_interest,
                                           num_workers=args.n_processes_patching)
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
        device=torch.device('cuda', gpu_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # get inference parameters
    predictor_parameters_list = args.inference_inst_dict['predict_param_list']
    ensemble_flag = args.inference_inst_dict['ensemble_flag']
    postprocessing_param_dict = args.inference_inst_dict['postprocessing_dict']

    # initialize list of prediction output caches
    prediction_output_caches = []

    # iterate predictor parameter list
    for i, pred_parm_dict in enumerate(predictor_parameters_list):
        # status message
        print(f'\nRun inference for model {pred_parm_dict["config"]} [{i+1}/{len(predictor_parameters_list)}]...\n')
        # create output folder in cache for model i
        pred_cache_folder = os.path.join(path_to_cache, f'pred_{i}')
        if not os.path.exists(pred_cache_folder):
            os.makedirs(pred_cache_folder)
        prediction_output_caches.append(pred_cache_folder)

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(path_to_nnunet_results, full_dataset_name, f'nnUNetTrainer__{pred_parm_dict["plan"]}__{pred_parm_dict["config"]}'),
            use_folds=pred_parm_dict['folds'],
            checkpoint_name='checkpoint_final.pth',
        )

        # variant 1: give input and output folders
        predictor.predict_from_files(join(path_to_input_cache),
                                     join(pred_cache_folder),
                                     save_probabilities=False, overwrite=False,
                                     num_processes_preprocessing=args.n_processes_preprocessing,
                                     num_processes_segmentation_export=args.n_processes_saving,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    # check if ensemble is desired
    if ensemble_flag:
        # status message
        print(f'\nRun ensemble...\n')
        # create ensemble output folder
        ensemble_output_folder = os.path.join(path_to_cache, 'ensemble')
        if not os.path.exists(ensemble_output_folder):
            os.makedirs(ensemble_output_folder)

        # run ensembling
        ensemble_folders(prediction_output_caches, ensemble_output_folder, num_processes=args.n_processes_preprocessing)
    else:
        # set ensemble output folder to first folder in prediction output caches
        ensemble_output_folder = prediction_output_caches[0]

    # check if postprocessing is desired and possible
    if postprocessing_param_dict:
        # check if postprocessing pickle file exists
        if not os.path.exists(os.path.join(path_to_nnunet_results, postprocessing_param_dict['pp_pkl_path'])):
            # set postprocessing parameter dict to non and print status message that postprocessing is not possible
            postprocessing_param_dict = None
            print('Postprocessing is not possible because the pickle file does not exist.')
    # check if running postprocessing is desired and possible
    if postprocessing_param_dict:
        # status message
        print(f'Run postprocessing...')
        # load postprocessing pickle files
        pp_fns, pp_fn_kwargs = load_pickle(os.path.join(path_to_nnunet_results, postprocessing_param_dict['pp_pkl_path']))
        # run postprocessing
        apply_postprocessing_to_folder(ensemble_output_folder,
                                       path_to_output_cache,
                                       pp_fns,
                                       pp_fn_kwargs,
                                       os.path.join(path_to_nnunet_results, postprocessing_param_dict['plans_json_path']),
                                       num_processes=args.n_processes_preprocessing)
    else:
        # set cache output folder to ensemble output folder
        path_to_output_cache = ensemble_output_folder

    gc.collect()
    # print status message
    print(f'Load generated mask...')
    # remove unnecessary variables from memory
    # stitch segmentation mask and load it
    data_array = processor.stitch_segmentation_mask(output_cache_path=path_to_output_cache)
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
    # remove cache directory
    if not debug_flag:
        shutil.rmtree(path_to_cache)
    # remove deleted variable from memory
    gc.collect()
    # print status message
    print(f'Save results as OME TIFF files at "{args.output}"...')

    # print status message
    print(f'Finished mask generation for {args.input}!')
