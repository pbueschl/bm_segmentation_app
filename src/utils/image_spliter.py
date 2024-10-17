import numpy as np
import h5py
import tifffile as tif
import os
import nibabel as nib
from multiprocessing import Pool, cpu_count


def write_nnUNet_training_nifti_files(path_output_directory, data_array, resolution, image_nr, channel_nr, case_id):
    # define nifti file name
    path_to_nifti_file = os.path.join(path_output_directory, f'{case_id}_{image_nr:03d}_{channel_nr:04d}.nii.gz')
    # transpose data array to have XYZ order
    data_array = data_array.transpose((2, 1, 0))
    # Create a NIfTI image header with the appropriate dimensions and spacing information
    affine = np.array([[resolution[2], 0, 0, 0], [0, resolution[1], 0, 0], [0, 0, resolution[0], 0], [0, 0, 0, 1]])
    # create nifti image object
    nii_image = nib.Nifti1Image(data_array, affine=affine)
    # save the nifti file
    nib.save(nii_image, path_to_nifti_file)


def save_data_channels_for_inference(image_data_array, metadata_dict, output_path, image_id=0):
    # read voxel size from metadata
    voxel_size = metadata_dict['voxel_size']
    # if voxel_size is not of type dict, convert it to type dict
    if type(voxel_size) is not dict:
        voxel_size = eval(voxel_size)
    # convert voxel size from dict to list of order (Z,Y,X)
    resolution = [voxel_size['Z'], voxel_size['Y'], voxel_size['X']]

    # iterate channels
    for i in range(image_data_array.shape[1]):
        # save channels in nifti files
        write_nnUNet_training_nifti_files(output_path, image_data_array[:, i, :, :], resolution, image_id, i, 'infimg')


def select_channels(data_array, metadata, channels_of_interest_dict):
    """
    Returns a subset of the passed channels of the data array. The selection is based on the passed metadata and
    channels of interest
    :param data_array: source image data (numpy.ndarray)
    :param metadata: metadata of the source file (dict)
    :param channels_of_interest_dict: channels that should be returned in form of a dict holding lists of possible
                                      channel names (value) for a given unified channel name (key) (dict)
    :return:data array containing the requested channels (numpy.ndarray)
    """
    # get channel_list from metadata
    available_channels_list = metadata['channel_names']
    # if available channels list is not of type list, convert it to type list
    if type(available_channels_list) is not list:
        available_channels_list = eval(available_channels_list)

    # initialize empty list for indices of channels of interest
    channels_of_interest_indices_list = []
    # iterate channels of interest
    for unified_channel_name, possible_channel_names_list in channels_of_interest_dict.items():
        # iterate list of possible channel names
        for channel_name in possible_channel_names_list:
            # initialize error flag with 1
            error_flag = 1
            # check if possible channel name is present in passed metadata
            if any(channel_name == ch for ch in available_channels_list):
                # get index of channel
                channels_of_interest_indices_list.append(available_channels_list.index(channel_name))
                # reset error flag
                error_flag = 0
                # stop for loop
                break
            # check if error flag is set
        if error_flag:
            # if for loop was not stopped raise an error, that channel of interest is not present in the passed data.
            raise ValueError(f'The expected channel "{unified_channel_name}" does not seem to exist in the passed data'
                             f'file. Please check the data and/or the channel naming convention of the source file.')
    # return data array of selected channels
    return data_array[:, channels_of_interest_indices_list, :, :]


class IMSProcessor:
    def __init__(self, path_to_input_ims_file, path_to_cache, max_pixels, channels_of_interest_dict, overlap=0.1, num_workers=8):
        self.tile_info_dict = None
        self.number_of_overlapping_pixels = None
        self.tile_division_factors = None
        self.path_to_input_data = path_to_input_ims_file
        self.path_to_cache = path_to_cache
        self.path_to_cache_input = os.path.join(path_to_cache, 'input')
        self.path_to_cache_output = os.path.join(path_to_cache, 'output')
        self.channels_of_interest_dict = channels_of_interest_dict
        self.input_metadata = None
        self.channel_name_to_ims_identifier_dict = None
        self.read_metadata_from_ims_file()
        self.max_pixels = max_pixels
        self.overlap = overlap
        self.num_tiles = 0
        self.number_of_tiles = 1
        self.num_workers = num_workers
        self.image_shape = (self.input_metadata["image_size"]['Z'],
                            len(channels_of_interest_dict),
                            self.input_metadata["image_size"]['Y'],
                            self.input_metadata["image_size"]['X'])
        self.tile_size = None

    def read_metadata_from_ims_file(self):
        """
        Reads image data together with relevant metadata from an Imaris ims file and returns the image data together
        with the extracted metadata.

        :param path_to_ims_file: path to the ims file from which the data should be extracted (string)
        :return: image_data_array (numpy.ndarray), metadata_dict (dict)
        """
        # print status message
        print(f'Read "{self.path_to_input_data}" ...')
        # open ims file
        with h5py.File(self.path_to_input_data, 'r') as f:
            # generate list of available channels
            channel_list = [ch_id for ch_id in f['DataSetInfo'] if
                            ch_id.startswith('Channel') and 'Name' in f['DataSetInfo'][ch_id].attrs.keys()]

            channel_names = []

            # iterate channels
            for channel in channel_list:
                # read channel names
                ch_name_temp = f['DataSetInfo'][channel].attrs['Name'].tobytes().decode('ascii', 'ignore')
                channel_names.append(ch_name_temp)
                # check if channel is present in the list of possible name variants of the channels of interest
                for k, v in self.channels_of_interest_dict.items():
                    if ch_name_temp in v:
                        # replace list of possible name variants with used name
                        self.channels_of_interest_dict[k] = ch_name_temp

            # check if one of the channels of interest dictionary's values still holds a list of possible name variants
            for k, v in self.channels_of_interest_dict.items():
                if isinstance(v, list):
                    raise TypeError(f'''The channel "{k}" seems to be absent in the passed ims data file. 
                                        Please re-check the ims file for the missing or misspelled channel (valid name variants are: "{', '.join(v)}").''')

            # initialize empty dict for voxel and image sizes
            voxel_size = {}
            image_size = {}
            # read min and max metric coordinates as well as pixel size in all three dimensions and calculate voxel size
            for i, d in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
                # read the highest metrical coordinate
                max_coord = float(f['DataSetInfo']['Image'].attrs[f'ExtMax{i}'].tobytes().decode('ascii', 'decode'))
                # read the lowest metrical coordinate
                min_coord = float(f['DataSetInfo']['Image'].attrs[f'ExtMin{i}'].tobytes().decode('ascii', 'decode'))
                # read the pixel size
                pixel_size = int(f['DataSetInfo']['Image'].attrs[d].tobytes().decode('ascii', 'decode'))

                # calculate metrical voxel size
                voxel_size[d] = (max_coord - min_coord) / pixel_size

                # add dimension size to image size dict
                image_size[d] = pixel_size

                # add dictionary that translates channel names to channel identifiers of the ims file as argument
                self.channel_name_to_ims_identifier_dict = {
                    f['DataSetInfo'][channel].attrs['Name'].tobytes().decode('ascii', 'ignore'): channel for channel in
                    channel_list}

        # generate meta data dict
        metadata_dict = {'axes': 'ZCYX',
                         'axes_info': 'ZCYX',
                         'channels': len(channel_names),
                         'slices': image_size['Z'],
                         'hyperstack': True,
                         'mode': 'grayscale',
                         'Channel': [{'Name': s.lower()} for s in channel_names],
                         'image_size': image_size,
                         'voxel_size': voxel_size,
                         'PhysicalSizeX': voxel_size['X'],
                         'PhysicalSizeXUnit': 'um',
                         'PhysicalSizeY': voxel_size['Y'],
                         'PhysicalSizeYUnit': 'um',
                         'PhysicalSizeZ': voxel_size['Z'],
                         'PhysicalSizeZUnit': 'um',
                         'original_file': os.path.basename(self.path_to_input_data)}

        # return dictionary with relevant metadata
        self.input_metadata = metadata_dict

    def closest_factors(self, n):
        """
        Returns two factors of n that are as close as possible.
        """
        for i in range(int(n ** 0.5), 0, -1):
            if n % i == 0:
                return i, n // i

    def split_image(self):
        # calculate number of total pixels
        number_of_pixels = np.prod(self.image_shape)

        # check how many tiles are needed
        number_of_tiles = np.ceil(number_of_pixels / self.max_pixels).astype(int)
        # initialize utility dict and list
        factor_dict = {}
        factor_ratio_list = []
        # test different factors
        for i in range(3):
            factor_dict[number_of_tiles + i] = self.closest_factors(number_of_tiles + i)
            # check ratio
            factor_ratio_list.append(
                max(factor_dict[number_of_tiles + i]) / min(factor_dict[number_of_tiles + i]))
        # define final number of tiles
        number_of_tiles += factor_ratio_list.index(min(factor_ratio_list))
        # define division factors in x and y dimension
        if self.image_shape[3] > self.image_shape[2]:
            division_factors = factor_dict[number_of_tiles]
            #if self.image_shape[2] * 2 > self.image_shape[3]:
            #    division_factors = (2, np.ceil(number_of_tiles / 2).astype(int))
            #else:
            #    division_factors = (1, number_of_tiles)

        else:
            division_factors = factor_dict[number_of_tiles][::-1]
            #if self.image_shape[3] * 2 > self.image_shape[2]:
            #    division_factors = (np.ceil(number_of_tiles / 2).astype(int), 2)
            #else:
            #    division_factors = (number_of_tiles, 1)
        # add tile division factors to attributes
        self.tile_division_factors = division_factors
        # calculate tile size
        self.tile_size = np.ceil(np.array(self.image_shape[-2:]) / np.array(division_factors))
        # calculate overlap in any direction
        self.number_of_overlapping_pixels = np.ceil(self.tile_size * self.overlap / 2).astype(int)

        # divide image into tiles
        tiles_in_x_direction = division_factors[-1]
        tiles_in_y_direction = division_factors[-2]

        # load image array
        # image_array = self.__read_image_array()
        # reduce channels
        # image_array = select_channels(image_array, self.input_metadata, self.channels_of_interest_dict)
        # initialize tile id counter
        tile_id = 0
        # initialize tile information dict
        tile_info_dict = {}
        # Initialize task list for multiprocessing
        tasks = []
        # iterate tiles in y direction
        for y_tile_idx in range(tiles_in_y_direction):
            # define start index
            y_start_idx = max(int(y_tile_idx * self.tile_size[0] - self.number_of_overlapping_pixels[0]), 0)
            # define_stop_index
            y_stop_idx = min(int((y_tile_idx + 1) * self.tile_size[0] + self.number_of_overlapping_pixels[0]),
                             self.image_shape[2])

            # iterate tiles in x direction
            for x_tile_idx in range(tiles_in_x_direction):
                # define start index
                x_start_idx = max(int(x_tile_idx * self.tile_size[1] - self.number_of_overlapping_pixels[1]), 0)
                # define_stop_index
                x_stop_idx = min(int((x_tile_idx + 1) * self.tile_size[1] + self.number_of_overlapping_pixels[1]),
                                 self.image_shape[3])

                # update tile info dict
                tile_info_dict[tile_id] = {"y_start_idx": y_start_idx,
                                           "y_stop_idx": y_stop_idx,
                                           "x_start_idx": x_start_idx,
                                           "x_stop_idx": x_stop_idx}
                # Add the task to the list
                tasks.append((y_start_idx, y_stop_idx, x_start_idx, x_stop_idx, tile_id, number_of_tiles))

                # increment tile id
                tile_id += 1
        # Use multiprocessing to process tiles in parallel
        print('Number of workers: ', self.num_workers)
        with Pool(processes=min(self.num_workers, len(tasks))) as pool:
            pool.starmap(self.process_tile, tasks)
        # add tile info dict to attributes
        self.tile_info_dict = tile_info_dict
        self.num_tiles = len(tasks)

    def process_tile(self, y_start_idx, y_stop_idx, x_start_idx, x_stop_idx, tile_id, number_of_tiles):
        """
        Function to process and save a tile. To be used with multiprocessing.
        """
        # Status message
        print(f"Start processing of tile [{tile_id + 1:2d}/{number_of_tiles:2d}] ...")
        # Open the ims file within the worker process
        with h5py.File(self.path_to_input_data, 'r') as f:
            tile_array = np.zeros(
                (self.image_shape[0], self.image_shape[1], y_stop_idx - y_start_idx, x_stop_idx - x_start_idx),
                dtype=np.float32
            )

            # Iterate channels of interest
            for ch_idx, channel_name in enumerate(self.channels_of_interest_dict.values()):
                tile_array[:, ch_idx, :, :] = f['DataSet']['ResolutionLevel 0']['TimePoint 0'][
                                                  self.channel_name_to_ims_identifier_dict[channel_name]]['Data'][
                                              :self.image_shape[0], y_start_idx:y_stop_idx, x_start_idx:x_stop_idx]

        # Save tile for inference as a nifti file
        save_data_channels_for_inference(tile_array, self.input_metadata, self.path_to_cache_input, tile_id)
        # Status message
        print(f"... finished processing of tile [{tile_id + 1:2d}/{number_of_tiles:2d}].")

    def stitch_segmentation_mask(self):
        # get nifti files with the processed segmentations
        segmentation_nifti_file_list = sorted(
            [os.path.join(self.path_to_cache_output, f) for f in os.listdir(self.path_to_cache_output) if
             f.endswith('.nii.gz')])

        # create an array with ones of the size of the input image as basis for the stitched segmentation mask
        stitched_segmentation_mask = np.ones((self.image_shape[0], self.image_shape[2], self.image_shape[3]),
                                             dtype=bool)

        # iterate segmentation mask files and load the data arrays
        for i, file_name in enumerate(segmentation_nifti_file_list):
            # print status message
            print("Load segmentation mask ", file_name, " for stitching...")
            # load segmentation mask
            segmentation_mask_array = nib.load(file_name)
            segmentation_mask_array = segmentation_mask_array.get_fdata()
            # reshape mask array
            segmentation_mask_array = segmentation_mask_array.transpose((2, 1, 0)).astype(bool)
            # add segmentation mask array to stitched mask array
            stitched_segmentation_mask[:,
            self.tile_info_dict[i]["y_start_idx"]: self.tile_info_dict[i]["y_stop_idx"],
            self.tile_info_dict[i]["x_start_idx"]: self.tile_info_dict[i]["x_stop_idx"]] = np.logical_and(
                stitched_segmentation_mask[:,
                self.tile_info_dict[i]["y_start_idx"]: self.tile_info_dict[i]["y_stop_idx"],
                self.tile_info_dict[i]["x_start_idx"]: self.tile_info_dict[i]["x_stop_idx"]], segmentation_mask_array
            )

        # convert datatype of stitched segmentation mask to int and multiply it with a factor for better visualization
        stitched_segmentation_mask = stitched_segmentation_mask.astype(np.uint8) * 128

        # return the stitched segmentation mask
        return stitched_segmentation_mask.astype(np.uint8)


'''
# Usage:
processor = IMSProcessor("path_to_your_ims_file.ims", 63 * 2 * 1600 * 2700, channels_dict)
for chunk in processor.process_chunks:
    # Process each chunk here
    pass
'''
'''
path_to_input_cache = 'cache/input'
path_to_output_cache = 'cache/output'
if not os.path.exists(path_to_input_cache):
    os.makedirs(path_to_input_cache)
if not os.path.exists(path_to_output_cache):
    os.makedirs(path_to_output_cache)

# import src.utils.image_spliter as ims
path_input_tif_file= '/home/work/phd/projects/vessel_dapi_seg__nnunet/data/test_ng_20231004/cache/input_data_array.ome.tif'
path_to_cache = '/home/work/phd/projects/vessel_dapi_seg__nnunet/data/test_ng_20231004/cache'
max_pixels = 67*2*2000*1000
channels_of_interest = {
    'dapi': ['dapi'],
    'endomucin': ['endomucin']}
proc = ims.IMSProcessor(path_input_tif_file, path_to_cache, max_pixels, channels_of_interest)
proc.split_image()
print("ende")'''
