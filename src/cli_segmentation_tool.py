import argparse
from utils.inference import inference


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
    parser = argparse.ArgumentParser(
        description='Creates vessel/dapi mask for a given ims file and saves the result in '
                    'a OME TIFF file.')

    # add arguments for the input path and output directory
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input ims file')
    parser.add_argument('-o', '--output', required=True,
                        help='Directory for storing the OME TIFF file with image and mask data')
    parser.add_argument('-m', '--mask_selection', required=True,
                        default='vessels',
                        help='Select the desired mask that should be generated.')
    parser.add_argument('-c', '--channel_selection',
                        help='Select the desired channel of vessel staining that should be used in addition to the DAPI channel.')
    # parse the arguments
    args = parser.parse_args()

    # transform the selected mask to the according dataset id
    args.dataset_id = mask_dataset_id_dict[args.mask_selection]
    # modify channels_of_interest dict
    channels_of_interest = {
        'dapi': channels_of_interest['dapi'],
        args.channel_selection: channels_of_interest[args.channel_selection],
    }
    # call inference function to generate desired mask
    inference(args, channels_of_interest)


if __name__ == "__main__":
    main()
