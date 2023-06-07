import argparse
from utils.inference import inference


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
    inference(args, channels_of_interest)


if __name__ == "__main__":
    main()
