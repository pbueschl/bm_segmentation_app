import argparse
import os
import shutil
import tifffile as tif
import numpy as np
import nibabel as nib
from utils import ims_files, utils
import subprocess


def main():
    # define needed channels
    channels_of_interest = {
        'dapi': ['dapi'],
        'endomucin': ['endomucin'],
        'endoglin': ['endoglin', 'endoglin_bad']
    }
    # create the parser object
    parser = argparse.ArgumentParser(
        description='Creates vessel/dapi mask for a given ims file and saves the result in '
                    'a OME TIFF file.')

    # add arguments for the input path and output directory
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input ims file')
    parser.add_argument('-o', '--output', required=True,
                        help='Directory for storing the OME TIFF file with image and mask data')
    parser.add_argument('-d', '--dataset_id', required=True,
                        help='Enter nnUNet dataset id for either "vessel" or "dapi" mask model.')
    # parse the arguments
    args = parser.parse_args()

    # call inference function to generate desired mask
    inferece(args, channels_of_interest)

def inferece(args, channels_of_interest):
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
    # construct command to run nnUNet_predict script
    inference_command = f' export nnUNet_raw="/home/paul/projects/vessel_seg__nnunet/data/nnUNet_raw";\
                           export nnUNet_preprocessed="/home/paul/projects/vessel_seg__nnunet/data/nnUNet_preprocessed";\
                           export nnUNet_results="/home/paul/projects/vessel_seg__nnunet/data/nnUNet_results";\
                            . /home/paul/anaconda3/bin/activate; \
                            conda activate nnunet; \
                            nnUNetv2_predict -i {path_to_input_cache} -o {path_to_output_cache} -d {args.dataset_id} -c 3d_fullres'
    # run inference
    subprocess.run(inference_command, shell=True, check=True, executable='/bin/bash')

    # print status message
    print(f'Load generated mask...')
    # load generated mask
    img = nib.load('cache/output/infimg_000.nii.gz')
    predicted_mask_array = img.get_fdata()

    # reshape predicted mask array and change boolen values to higher ones for better presentability
    predicted_mask_array = predicted_mask_array.transpose((2, 1, 0)) * 128
    # expand dimension of predicted mask array from ZYX to ZCYX
    predicted_mask_array = np.expand_dims(predicted_mask_array, axis=1).astype(np.uint8)
    # delete cache directory and files ToDo:
    #shutil.rmtree('cache')
    # concatenate image data with the generated mask ToDo
    output_data_array = np.concatenate((input_data_array, predicted_mask_array), axis=1)
    # output_data_array = np.concatenate((inference_data_array, predicted_mask_array), axis=1)

    # read list of channel names from metadata
    channel_names = metadata['channel_names']
    # convert channel names to list if it is not already of type list
    if type(channel_names) is not list:
        channel_names = eval(channel_names)
    # add predicted mask to channel names ToDo:
    # channel_names = list(channels_of_interest.keys())
    channel_names.append('predicted_mask_vessels')
    # update metadata
    metadata['channel_names'] = channel_names
    metadata['channels'] = len(channel_names)

    # define output file name
    path_to_output_ome_tiff_file = os.path.join(args.output,
                                                f'{os.path.splitext(os.path.basename(args.input))[0]}.ome.tif')
    # print status message
    print(f'Save concatenated data array as OME TIFF file at "{path_to_output_ome_tiff_file}"...')
    # save resulting ome tiff file
    tif.imwrite(path_to_output_ome_tiff_file,
                output_data_array,
                shape=output_data_array.shape,
                imagej=True,
                metadata=metadata)

    # print status message
    print(f'Finished mask generation for {args.input}!')


if __name__ == "__main__":
    main()
