import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class CuteDataset(Dataset):
    
    def __init__(self, img_list, img_dir, mask_dir, transform_mask=None,
                 transform_img=None, train=True):
        
        super(CuteDataset, self).__init__()
        self.img_list = img_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.train = train
        
    def __len__(self):
        
        return(len(self.img_list))
    
    def __getitem__(self, index):
        
        img_path = os.path.join(self.img_dir,self.img_list[index])
        mask_path = os.path.join(self.mask_dir,
                                 self.img_list[index].replace(".jpg", ".png"))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        
        if self.transform_img:
            
            img = self.transform_img(image)
        
        if self.transform_mask:
            
            mask = self.transform_mask(mask)
            
        return {"image": img, "mask": mask}
        


