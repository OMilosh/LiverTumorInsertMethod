from Generation.utils import get_healthy_liver_numbers, read_volume_mask
from Generation.inserting import generate_tumor, randomize_paramerters

import nibabel as nib
import numpy as np
import argparse
import random
import os
import pandas as pd
from tqdm import tqdm

def generate(healthy_liver_num, tumor_liver_list,neighbour_processing=False):
    img_volume_h, img_seg_h = read_volume_mask(f"../separate_tumors/volume_healthy/volume-{healthy_liver_num}.nii",
                                               f"../separate_tumors/mask_healthy/segmentation-{healthy_liver_num}.nii")
    for tumor in tumor_liver_list:
        img_volume_t, img_seg_t = read_volume_mask(f"../separate_tumors/volume/volume-{tumor}.nii",
                                                   f"../separate_tumors/mask/segmentation-{tumor}.nii")
        tumor_index = random.randint(2, img_seg_t.max())
        if neighbour_processing:
            df = pd.read_csv('../out.csv')
            tumor_info = df[(df["liver_num"]==tumor) & (df["tumor_num"]==tumor_index)]
            tumor_info = tumor_info[['tumor_mean', 'neighbour_mean']]
            tumor_neighbour_diff = tumor_info["neighbour_mean"] - tumor_info["tumor_mean"]
            tumor_neighbour_diff = tumor_neighbour_diff.values[0]
        else:
            tumor_neighbour_diff = None
        img_seg_t = np.where(img_seg_t==tumor_index, 1, 0)
        tmp_new_volume, tmp_liver_and_tumor_mask = generate_tumor(img_seg_h, img_volume_h, 
                                                              img_seg_t, img_volume_t, )
        if tmp_liver_and_tumor_mask.shape[0] != 0:
            img_volume_h, img_seg_h = tmp_new_volume, tmp_liver_and_tumor_mask
        else:
            continue
        tmp_new_volume, tmp_liver_and_tumor_mask = None, None
    return img_volume_h, img_seg_h

parser = argparse.ArgumentParser()

parser.add_argument('--num', type=int, default="num")
parser.add_argument('--neighbours', action='store_true')

def main():
    args = parser.parse_args()
    num = args.num
   
    for _ in tqdm(range(num)):
        healthy_liver_num, tumor_liver_list = randomize_paramerters()

        img_volume, img_seg = generate(healthy_liver_num, tumor_liver_list, args.neighbours)
        
        empty_header = nib.Nifti1Header()
        affine = nib.load(f"../separate_tumors/volume_healthy/volume-{healthy_liver_num}.nii").affine
        img_seg[img_seg>=2] = 2
        img_volume = nib.Nifti1Image(img_volume, affine, empty_header)
        img_seg = nib.Nifti1Image(img_seg, affine, empty_header)
        
        num_files = len(os.listdir(f'../datasets/generated/InsertMethod/volume/'))
        nib.save(img_volume, f'../datasets/generated/InsertMethod/volume/volume-{num_files}.nii')
        nib.save(img_seg, f'../datasets/generated/InsertMethod/mask/segmentation-{num_files}.nii')

        
        
if __name__ == '__main__':
    main()
