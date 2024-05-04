import os
import numpy as np
import torch
from PIL import Image

def convert_RGB_to_H_or_E(oriImg,target_H_main_folder, target_E_main_folder,bacth_img,img_size):
    # image = np.asarray(Image.open(os.path.join(main_folder,subfolder,filename)))#(height, width, channel)int 
    #image = np.transpose(image, (2, 0, 1))#change to (channel, height, width)
    # image = image.astype('float32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_H_tensor = torch.empty((0, img_size,img_size), dtype=torch.float32).to(device)
    all_E_tensor = torch.empty((0,img_size,img_size),dtype=torch.float32).to(device)
    for batch_i in range(bacth_img): #16
        image = oriImg[batch_i,:,:,:] #float32
        image_ones=np.ones(image.shape,dtype=np.float32)
        image_ones=torch.from_numpy(image_ones).to(device)
        image = torch.maximum(image,image_ones)#change all pixel intensity =0 to =1
        width=image.size(2)
        height= image.size(1)
        
        RGB_absorption=torch.log10(255.0*(image_ones/image))
        #If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
        gray_image_H=torch.matmul(RGB_absorption.permute(1,2,0),torch.tensor([1.838, 0.0341, -0.76]).to(device))
        gray_image_H=torch.clip(gray_image_H,0.0,1.0)
        gray_image_H=255.0*gray_image_H
        gray_image_H = gray_image_H.type(torch.uint8)
        gray_image_H = torch.unsqueeze(gray_image_H,0)
        # gray_image_H = gray_image_H.permute(1,2,0)
        all_H_tensor = torch.cat((all_H_tensor, gray_image_H), 0)
        
        gray_image_E=torch.matmul(RGB_absorption.permute(1,2,0),torch.tensor([-1.373, 0.772, 1.215]).to(device))
        gray_image_E=torch.clip(gray_image_E,0.0,1.0)
        gray_image_E=255.0*gray_image_E
        gray_image_E = gray_image_E.type(torch.uint8)
        gray_image_E = torch.unsqueeze(gray_image_E,0)
        # gray_image_E = gray_image_E.permute(1,2,0)
        all_E_tensor = torch.cat((all_E_tensor, gray_image_E), 0)
        
        
        # img_H = Image.fromarray(np.squeeze(gray_image_H.cpu().numpy().astype(np.uint8),axis=2)) #224,224
        # # img_H.save(os.path.join(target_H_main_folder,sub_folder_index))
        # img_H.save(target_H_main_folder)

        # img_E = Image.fromarray(np.squeeze(gray_image_E.cpu().numpy().astype(np.uint8),axis=2))
        # img_E.save(target_E_main_folder)

    return all_H_tensor,all_E_tensor
