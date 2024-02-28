import numpy as np
from sklearn.cluster import DBSCAN

def merge_liver_tumor(liver_mask, tumor_mask):
    merged_liver_mask = liver_mask.copy()
    merged_liver_mask = np.where(tumor_mask!=0, tumor_mask, liver_mask)
    return merged_liver_mask

# separate tumor masks DBSCAN
def get_clusters_masks(img_seg):
    
    def get_separate_mask(img_seg, indexes):
        mask = np.zeros((img_seg.shape[0], img_seg.shape[1], img_seg.shape[2]))
        mask[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1.
        return mask
    
    array_tumors = np.asarray(np.zeros(img_seg.shape), dtype=np.uint8)
    
    X = np.stack(np.where(img_seg==2.), axis=-1)
    db = DBSCAN(eps=4, min_samples=10).fit(X)
    
    labels = db.labels_
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    for i, k in enumerate(unique_labels, start=1):
        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        array_tumors = np.where(get_separate_mask(img_seg, xy)==1, i, array_tumors)
        
    return np.asarray(array_tumors, dtype=np.uint8)