import argparse
import cli_segmentation_tool as segtool
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Converts Imaris ims files to OME TIFF files')
    parser.add_argument('-i', '--input', required=True, help='Path to input directory of ims files')
    parser.add_argument('-o', '--output', required=True, help='Path for storing the generated OME TIFF mask files')
    parser.add_argument('-v', '--vessel_mask', action='store_true', help='Create vessel mask')
    parser.add_argument('-t', '--tissue_mask', action='store_true', help='Create tissue mask')
    parser.add_argument('-vm', '--vessel_marker', type=str, default='endomucin',
                        help='Name of marker used for staining the vessels')
    args = parser.parse_args()

    return args


def main():
    # Parse arguments
    args = parse_args()
    # create list with masks that should be created
    masks_to_create = []
    if args.vessel_mask:
        masks_to_create.append('vessels')
    if args.tissue_mask:
        masks_to_create.append('tissue')

    # get list of ims files in the input directory
    ims_files = [f for f in os.listdir(args.input) if f.endswith('.ims')]

    # create separate output subdirectories for all masks that should be created
    for mask_type in masks_to_create:
        mask_output_dir = os.path.join(args.output, mask_type)
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)

    # iterate over ims files
    for f in ims_files:
        for mask_type in masks_to_create:
            arguments = ['--input', os.path.join(args.input, f),
                         '--output', os.path.join(args.output, mask_type),
                         '--mask_selection', mask_type,
                         '--channel_selection', args.vessel_marker]
            # Save the original sys.argv
            original_argv = sys.argv
            try:
                # Simulate passing command-line arguments
                sys.argv = ['segtool.main'] + arguments
                # Call the function
                segtool.main()
            except (TypeError, ValueError) as e:
                # create a txt file that with the name 'failed__{mask type}_mask_{ims file name without suffix}.txt'
                # and save it to the output directory
                with open(os.path.join(args.output, f'failed__{mask_type}_mask_{f[:-4]}.txt'), 'w') as file:
                    file.write(f'Failed to create {mask_type} mask for file:\n   ' + os.path.join(args.input, f) + '\n\nCached exception: \n' + str(e))
            finally:
                # Restore the original sys.argv
                sys.argv = original_argv


if __name__ == '__main__':
    main()
