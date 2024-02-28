import nibabel as nib
import cv2
import numpy as np


def change_spine_orientation(data_volume, data_mask):
    orient = nib.aff2axcodes(data_volume.affine)
    orient_arr = nib.orientations.axcodes2ornt(orient)
    img_volume = data_volume.get_fdata()
    img_volume= np.rot90(nib.orientations.apply_orientation(img_volume, orient_arr))
    img_mask = data_mask.get_fdata()
    img_mask= np.rot90(nib.orientations.apply_orientation(img_mask, orient_arr))
    return img_volume, img_mask

def make_color_mask(mask_gs,rgb=(100,0,190)):
    new_mask = np.expand_dims(mask_gs, axis = -1)
    red = new_mask * rgb[0]
    blue = new_mask * rgb[1]
    green = new_mask * rgb[2]
    return np.concatenate((red, blue, green), axis=-1) / 255
    
def UI_visualization(volume, tumor_liver_mask):
    
# creating window
    WINDOW_NAME = "output"
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,1400,700)
    
# creating trackbars
    cv2.createTrackbar("slice",WINDOW_NAME,0,volume.shape[2] - 1,lambda x:x)
#     cv2.createTrackbar("full | only liver | only tumor",WINDOW_NAME,0,2,lambda x:x)
    if tumor_liver_mask.max() > 2:
        cv2.createTrackbar("merged tumors->0 | single tumor index",WINDOW_NAME,0,tumor_liver_mask.max()-1,lambda x:x)
    cv2.createTrackbar("alpha volume",WINDOW_NAME,20,100,lambda x:x)
    cv2.createTrackbar("alpha tumor",WINDOW_NAME,500,1000,lambda x:x)
    if 1 in tumor_liver_mask:
        cv2.createTrackbar("alpha liver",WINDOW_NAME,500,1000,lambda x:x)
        
# start showing window
    while True:
        # get parameters    
#         type_ui = cv2.getTrackbarPos("full | only liver | only tumor",WINDOW_NAME)
        slice_num = cv2.getTrackbarPos("slice",WINDOW_NAME)
        alpha_volume = cv2.getTrackbarPos("alpha volume",WINDOW_NAME)/10000
        alpha_tumor = cv2.getTrackbarPos("alpha tumor",WINDOW_NAME)/1000
        if tumor_liver_mask.max() > 2:
            tumor_index = cv2.getTrackbarPos("merged tumors->0 | single tumor index", WINDOW_NAME)

        # set full slice volume 
        volume_sl = volume[:,:,slice_num]

        # liver mask or liver volume
        alpha_liver = cv2.getTrackbarPos("alpha liver",WINDOW_NAME)/1000
        liver_mask_sl = tumor_liver_mask[:,:,slice_num].copy()
        liver_mask_sl[liver_mask_sl!=0]=1
#       liver_mask_sl = np.where(tumor_liver_mask[:,:,slice_num]!=0, 1, 0)
        liver_mask_sl = make_color_mask(liver_mask_sl,rgb=(0,130,30))
        # showing only liver volume
#         if type_ui == 1:
#             volume_sl = np.where(tumor_liver_mask[:,:,slice_num]!=0, volume_sl, 0)  
        
        # tumor preparation
        tumor_mask_sl = tumor_liver_mask[:,:,slice_num].copy()
        if tumor_liver_mask.max() > 2:
            # all tumors
            if tumor_index == 0:
                tumor_mask_sl[tumor_mask_sl<=1] = 0
                tumor_mask_sl[tumor_mask_sl>1] = 1

            # single tumor from tumors list 
            else:
                tumor_mask_sl[tumor_mask_sl!=tumor_index+1]=0
                tumor_mask_sl[tumor_mask_sl==tumor_index+1]=1
        else:
            tumor_mask_sl[tumor_mask_sl!=2]=0
            tumor_mask_sl[tumor_mask_sl==2]=1

        # showing only tumor/s                         
#         if type_ui == 2:
#             volume_sl = np.where(tumor_mask_sl==1., volume_sl, 0)
        volume_sl = make_color_mask(volume_sl,rgb=(255,255,255))
        tumor_mask_sl = make_color_mask(tumor_mask_sl,rgb=(100,0,190))
   
        # get masked image         
        img_with_mask = cv2.addWeighted(volume_sl, alpha_volume, tumor_mask_sl, alpha_tumor, 0)
        img_with_mask = cv2.addWeighted(img_with_mask, 1, liver_mask_sl, alpha_liver, 0) 
 
        cv2.imshow("output",img_with_mask)
        c = cv2.waitKey(1)
        if c == ord('q'):
            cv2.destroyAllWindows()
            break
            
