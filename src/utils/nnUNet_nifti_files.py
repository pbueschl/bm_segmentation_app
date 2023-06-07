import argparse
import os
import json
import nibabel as nib
import numpy as np


def set_up_nnUNet_file_structure(path_to_nnUNet_dataset, dataset_id):
    # combine path to nnUNet dataset
    path_to_nnUNet_dataset = os.path.join(path_to_nnUNet_dataset, dataset_id)
    # check if output directory exists, if not create it
    if not os.path.exists(path_to_nnUNet_dataset):
        os.makedirs(path_to_nnUNet_dataset)

    # define path to nnUNet typic imagesTr directory
    path_to_images_tr = os.path.join(path_to_nnUNet_dataset, 'imagesTr')
    # check if output subdirectory imagesTr exists
    if not os.path.exists(path_to_images_tr):
        os.makedirs(path_to_images_tr)
    # define path to nnUNet typic labelsTr directory
    path_to_labels_tr = os.path.join(path_to_nnUNet_dataset, 'labelsTr')
    # check if output subdirectory labelsTr exists
    if not os.path.exists(path_to_labels_tr):
        os.makedirs(path_to_labels_tr)
    # return
    return path_to_nnUNet_dataset, path_to_images_tr, path_to_labels_tr


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


def write_nnUNet_label_nifti_files(path_output_directory, data_array, resolution, image_nr, case_id):
    # define nifti file name
    path_to_nifti_file = os.path.join(path_output_directory, f'{case_id}_{image_nr:03d}.nii.gz')
    # transpose data array to have XYZ order and convert it to uint8
    data_array = data_array.transpose((2, 1, 0)).astype(np.uint8)
    # Create a NIfTI image header with the appropriate dimensions and spacing information
    affine = np.array([[resolution[2], 0, 0, 0], [0, resolution[1], 0, 0], [0, 0, resolution[0], 0], [0, 0, 0, 1]])
    # create nifti image object
    nii_image = nib.Nifti1Image(data_array, affine=affine)
    # save the nifti file
    nib.save(nii_image, path_to_nifti_file)
