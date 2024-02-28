import nibabel as nib
import numpy as np
from .visualization import change_spine_orientation
from skimage.segmentation import expand_labels


def read_volume_mask_orig(path_volume, path_mask):
    data_seg = nib.load(path_mask)
    data_volume = nib.load(path_volume)
    img_volume, img_seg  = change_spine_orientation(data_volume, data_seg)
    z_liver = np.unique(np.argwhere(img_seg==1)[:,-1])
    img_seg = img_seg[:,:,z_liver]
    img_volume = img_volume[:,:,z_liver]
    return img_volume, img_seg

def read_volume_mask(path_volume, path_mask):
    data_seg = nib.load(path_mask)
    data_volume = nib.load(path_volume)
    img_seg  = data_seg.get_fdata().astype(np.uint8)
    img_volume = data_volume.get_fdata().astype(np.int32)
    z_liver = np.unique(np.argwhere(img_seg==1)[:,-1])
    img_seg = img_seg[:,:,z_liver]
    img_volume = img_volume[:,:,z_liver]
    return img_volume, img_seg

def get_healthy_liver_numbers():
    return [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]

def calc_neighbours(mask_tumor, mask_liver, volume):
    expanded = expand_labels(mask_tumor, distance=5) - mask_tumor
    expanded[mask_liver==0] = 0
    return np.mean(volume[expanded==1])
