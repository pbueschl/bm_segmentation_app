from utils.inference import inference
from gooey import Gooey, GooeyParser


@Gooey(advanced=True)
def main():
    # define needed channels
    channels_of_interest = {
        'dapi': ['dapi'],
        'endomucin': ['endomucin'],
        'endoglin': ['endoglin', 'endoglin_bad']
    }
    # define mask dataset dict, assigns each mask type a dataset
    mask_dataset_id_dict = {
        'vessels': 4,
        'tissue': 0
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
    parser.add_argument('-d', '--dataset_id', required=True, choices=['vessels', 'tissue'], widget="Dropdown",
                        default='vessels',
                        help='Select the desired mask that should be generated.')
    # parse the arguments
    args = parser.parse_args()

    # transform the selected mask to the according dataset id
    args.dataset_id = mask_dataset_id_dict[args.dataset_id]

    # call inference function to generate desired mask
    inference(args, channels_of_interest)


if __name__ == "__main__":
    main()
