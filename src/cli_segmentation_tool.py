import argparse
from utils.inference_new import inference
from utils.utils import get_used_nnunet_folds_configuration_and_plan
import pathlib
import time


def main():
    # define needed channels
    channels_of_interest = {
        'dapi': ['dapi', 'DAPI', 'Dapi'],
        'endomucin': ['endomucin', 'ENDOMUCIN', 'Endomucin'],
        'endoglin': ['endoglin', 'endoglin_bad']
    }
    # define mask dataset dict, assigns each mask type a dataset
    mask_dataset_id_dict = {
        'vessels': 8,
        'tissue': 160,
        'dilated_vessels': 400
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
    parser.add_argument('-m', '--mask_selection',
                        default='vessels',
                        help='Select the desired mask that should be generated.')
    parser.add_argument('-c', '--channel_selection',
                        help='Select the desired channel of vessel staining that should be used in addition to the DAPI channel.')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='Select the GPU ID for inference.')
    parser.add_argument('-np', '--n_processes_preprocessing', type=int, default=1,
                        help='Define number of processes used for preprocessing.')
    parser.add_argument('-ns', '--n_processes_saving', type=int, default=1,
                        help='Define number of processes used for saving output files.')
    parser.add_argument('-ni', '--n_processes_patching', type=int, default=16,
                        help='Define number of processes used for patching the image.')
    # parse the arguments
    args = parser.parse_args()

    # transform the selected mask to the according dataset id
    args.dataset_id = mask_dataset_id_dict[args.mask_selection]
    # get folds, configuration and trainer
    args.nnunet_folds, args.nnunet_config, args.nnunet_plans = get_used_nnunet_folds_configuration_and_plan(args.dataset_id)

    # modify channels_of_interest dict
    if args.mask_selection == 'vessels':
        channels_of_interest = {
            'dapi': channels_of_interest['dapi'],
            args.channel_selection: channels_of_interest[args.channel_selection],
        }
    if args.mask_selection == 'tissue':
        channels_of_interest = {
            'dapi': channels_of_interest['dapi'],
            args.channel_selection: channels_of_interest[args.channel_selection],
        }

    if args.mask_selection == 'dilated_vessels':
        channels_of_interest = {
            'dapi': channels_of_interest['dapi'],
            args.channel_selection: channels_of_interest[args.channel_selection],
        }

    # call inference function to generate desired mask
    inference(args, channels_of_interest, gpu_id=args.gpu_id)


if __name__ == "__main__":
    start_time = time.time()
    main()
    # Get execution time in hours minutes and seconds
    time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Total execution time: {time_str}")
