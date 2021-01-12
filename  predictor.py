import os 
import sys
import time 

import cv2
import torch 
import dlib
import numpy as np
from PIL import Image
from argparse import Namespace
import torchvision.transforms as transforms


from models.psp import pSp
from utils.common import tensor2im
from scripts.align_all_parallel import align_face



class Predictor:
    def __init__(self, 
                    model_path, 
                    landmark_predictor_path, 
                    save_res_path, 
                    latent_dirs_p):

        self.inference_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.landmark_prediction_path = landmark_predictor_path

        #load pretrained model 
        ckpt = torch.load(model_path, map_location='cpu')
        self.opts = ckpt['opts']
        self.opts['checkpoint_path'] = model_path
        self.opts['test_batch_size'] = 1
        
        if 'learn_in_w' not in self.opts:
            self.opts['learn_in_w'] = False

        self.opts = Namespace(**self.opts)
        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()

        #dlib face landmarks predictor
        self.predictor = dlib.shape_predictor(landmark_predictor_path)
        
        #StyleGan2 Latent Directions 
        self.age_direction = np.load(os.path.join(latent_dirs_p, 'age.npy')).astype('float32')
        self.smile_direction = np.load(os.path.join(latent_dirs_p, 'smile.npy')).astype('float32')
        self.gender_direction = np.load(os.path.join(latent_dirs_p, 'gender.npy')).astype('float32')

    def __base64_image_decode(self, base64_str):
        pass

    def __base64_image_encode(self, image):
        pass 

    def __align_image(self, image):
        aligned_image = align_face(image, self.predictor)
        aligned_image = aligned_image.convert('RGB')
        return aligned_image
    
    def __preprocess_input(self, image):
        aligned_image = self.__align_image(image)
        input_image = self.inference_transforms(image)
        return input_image

    def __get_latent_directions(self, image, net):
        inp_img = self.__preprocess_input(image)
        with torch.no_grad():
            enc_im, latent_dir = self.net(inp_img.to('cuda').float().unsqueeze(0),
                                            randomize_noise=False,
                                            return_latents=True)

            latenr_dir = latenr_dir.data.cpu().numpy()
            return latenr_dir

    def __move_latent_directions(self, type, coeff):
        pass

    def get_prediction(self, base64_image, type, id, coeff):
        pass


if __name__ == "__main__":
    pass