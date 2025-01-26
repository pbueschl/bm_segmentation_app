from utils.inference import inference
from utils.utils import get_used_nnunet_folds_configuration_and_plan
from gooey import Gooey, GooeyParser
import time
import os


@Gooey(advanced=True,
       image_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'icons',
                              'gooey_icons')
       )
def main():
    # define patch size reduction factor
    patch_size_reduction_factor = 0.25

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
        'dilated_vessels': 400,
        'dilated_tissue': 71
    }
    # create the parser object
    parser = GooeyParser(
        description='Creates vessel/dapi mask for a given ims file and saves the result in '
                    'a OME TIFF file.')

    # add arguments for the input path and output directory
    parser.add_argument('-i', '--input', required=True, widget='FileChooser',
                        help='Path to input ims file')
    parser.add_argument('-o', '--output', required=True, widget='DirChooser',
                        help='Directory for storing the OME TIFF file with image and mask data')
    parser.add_argument('-m', '--mask_selection', required=True, choices=list(mask_dataset_id_dict.keys()),
                        widget="Dropdown",
                        default='vessels',
                        help='Select the desired mask that should be generated.')
    parser.add_argument('-c', '--channel_selection', required=True, choices=list(channels_of_interest.keys())[1:],
                        widget="Dropdown",
                        help='Select the desired channel of vessel staining that should be used in addition to the DAPI channel.')
    parser.add_argument(
        '--reduce_patch_size',  # Name of the argument
        action='store_const',  # Special action to store a constant value
        const=patch_size_reduction_factor,  # Value to store when the checkbox is checked
        default=None,  # Value when the checkbox is unchecked
        help=f'Reduce patch size by factor {int(1/patch_size_reduction_factor)}')
    # parse the arguments
    args = parser.parse_args()
    # set gpu id
    args.gpu_id = 0
    # set number of precesses used for preprocessing, saving and patching
    args.n_processes_preprocessing = 1
    args.n_processes_saving = 1
    args.n_processes_patching = 8

    # transform the selected mask to the according dataset id
    args.dataset_id = mask_dataset_id_dict[args.mask_selection]
    # get folds, configuration and trainer
    args.inference_inst_dict = get_used_nnunet_folds_configuration_and_plan(args.dataset_id)

    # modify channels_of_interest dict
    channels_of_interest = {
        'dapi': channels_of_interest['dapi'],
        args.channel_selection: channels_of_interest[args.channel_selection],
    }

    # call inference function to generate desired mask
    inference(args, channels_of_interest, gpu_id=args.gpu_id, tile_scaling_factor=args.tile_scaling_factor)


if __name__ == "__main__":
    start_time = time.time()
    main()
    # Get execution time in hours minutes and seconds
    time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Total execution time: {time_str}")
