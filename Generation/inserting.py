import random
import cv2
import numpy as np
from .clusterization import merge_liver_tumor
from .utils import get_healthy_liver_numbers, calc_neighbours

def get_bounds_mask(mask):
    location = np.where(mask==1)
    ymin, ymax = min(location[0]), max(location[0])
    xmin, xmax = min(location[1]), max(location[1])
    zmin, zmax = min(location[2]), max(location[2])
    return (ymin, ymax), (xmin, xmax), (zmin, zmax)

def get_mask_new_location(liver_and_tumor_mask, new_tumor_mask):
    
#     if there are already tumor in the liver: make sure liver mask == 1 
#     if it is just liver and liver mask >= 2 if its tumor
# new_tumor_mask max = 1
    
    (tumor_ymin, tumor_ymax), (tumor_xmin, tumor_xmax), (tumor_zmin, tumor_zmax) = get_bounds_mask(new_tumor_mask)
    cutted_tumor_mask =  new_tumor_mask[tumor_ymin:tumor_ymax+1, tumor_xmin:tumor_xmax+1, tumor_zmin:tumor_zmax+1]
    
    def random_point(liver_and_tumor_mask):
        (liver_ymin, liver_ymax), (liver_xmin, liver_xmax), (liver_zmin, liver_zmax) = get_bounds_mask(liver_and_tumor_mask)
        z = round(random.uniform(0.3, 0.7) * (liver_zmax - liver_zmin)) + liver_zmin

        coordinates = np.argwhere(liver_and_tumor_mask[:,:,z] == 1)
        random_index = np.random.randint(0, len(coordinates))
        yxz = coordinates[random_index].tolist() # get x,y
        yxz.append(z)
        return yxz
        
    def get_random_mask(cutted_tumor_mask, liver_and_tumor_mask):
        random_mask = np.zeros(liver_and_tumor_mask.shape,dtype=np.uint8)
        tumor_shape = cutted_tumor_mask.shape
        point = random_point(liver_and_tumor_mask)
        y_low, y_high = point[0], point[0] + tumor_shape[0]
        x_low, x_high = point[1], point[1] + tumor_shape[1]
        z_low, z_high = point[2], point[2] + tumor_shape[2]
        new_shape = random_mask[y_low:y_high, x_low:x_high, z_low:z_high].shape
        random_mask[y_low:y_high, x_low:x_high, z_low:z_high] = cutted_tumor_mask[:new_shape[0], :new_shape[1], :new_shape[2]]
        random_mask[liver_and_tumor_mask==0] = 0
        return random_mask
    
    return get_random_mask(cutted_tumor_mask, liver_and_tumor_mask)
  

def add_volume_mask_tumor(volume_healthy, volume_ill, tumor_mask_orig, tumor_mask_random):
#     tumor_mask_orig and tumor_mask_random max == 1 
    new_volume = volume_healthy.copy()
    new_volume[tumor_mask_random==1] = volume_ill[tumor_mask_orig==1][0:len(new_volume[tumor_mask_random==1])]
    return new_volume

def randomize_paramerters():
    healthy_num = random.choice(get_healthy_liver_numbers())
    num_tumors = random.randint(3,12)
    tumor_livers = [random.randint(0, 130) for _ in range(num_tumors)]
    tumor_livers = [i for i in tumor_livers if i not in get_healthy_liver_numbers()]
    return healthy_num, tumor_livers

# img_seg_t.max() == 1
def generate_tumor(liver_and_tumor_mask, volume_healthy, img_seg_t, img_volume_t, tumor_neighbour_diff=None):
    
    new_tumor_inside_liver = get_mask_new_location(liver_and_tumor_mask,
                                                    img_seg_t)
    volume_healthy = add_volume_mask_tumor(volume_healthy,
                                            img_volume_t,
                                            img_seg_t,
                                            new_tumor_inside_liver)
   
    if tumor_neighbour_diff is not None:
        neighbour_mean = calc_neighbours(new_tumor_inside_liver, np.where(liver_and_tumor_mask==1, 1, 0), volume_healthy)
        tmp = volume_healthy[new_tumor_inside_liver==1].copy()
        volume_healthy[new_tumor_inside_liver==1] = tmp/np.mean(tmp)*(neighbour_mean - tumor_neighbour_diff)
        tmp = None        
        
    liver_and_tumor_mask = merge_liver_tumor(liver_and_tumor_mask,
                                            np.where(new_tumor_inside_liver==1,
                                                    liver_and_tumor_mask.max()+1, 0))

    
    return volume_healthy, liver_and_tumor_mask