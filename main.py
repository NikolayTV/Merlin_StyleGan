import os
import sys
import time

sys.path.append('.')

import cv2
import dlib
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from argparse import Namespace
import torchvision.transforms as transforms

from models.psp import pSp
from utils.common import tensor2im
from scripts.align_all_parallel import align_face

PRETRAINED_WEIGHT = 'psp_ffhq_encode.pt'
TEST_DATA_DIR = './test_data'
RESULT_DATA_PATH = './test_data_res'
LATENT_DIRECTION_PATH = './latent_directions'
FACE_LANDMARKS_PATH = 'shape_predictor_68_face_landmarks.dat'
ALIGN_IMAGE = True
# COEFFS = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
COEFFS = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


INFERENCE_DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def main():
    test_images_paths = [os.path.join(TEST_DATA_DIR, img) for img in os.listdir(TEST_DATA_DIR) if
                         img.split('.')[1] == 'jpg' or img.split('.')[1] == 'png']
    num_images = len(test_images_paths)
    print('{} Images'.format(num_images))

    latent_dirs = []
    #load stylegan2 latent directions
    latent_dirs.append(np.load('./latent_directions/age.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/smile.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/gender.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/angle_horizontal.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/angle_pitch.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/beauty.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_angry.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_disgust.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_easy.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_fear.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_happy.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_sad.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/emotion_surprise.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/eyes_open.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/face_shape.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/glasses.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/height.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/race_black.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/race_white.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/race_yellow.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/smile.npy').astype('float32'))
    # latent_dirs.append(np.load('./latent_directions/width.npy').astype('float32'))

    latent_names = [['age'], ['smile'], ['gender'], ['angle_horizontal'], ['angle_pitch'], ['beauty'], ['emotion_angry'], ['emotion_disgust'], ['emotion_easy'],
            ['emotion_fear'], ['emotion_happy'], ['emotion_sad'], ['emotion_surprise'], ['eyes_open'], ['face_shape'], ['glasses'], ['height'],
            ['race_black'], ['race_white'], ['race_yellow'], ['smile'], ['width']]

    # Load pretraiden weights
    ckpt = torch.load(PRETRAINED_WEIGHT, map_location='cpu')
    # ckpt = torch.load(PRETRAINED_WEIGHT, map_location='cuda:0')

    opts = ckpt['opts']
    print(opts)

    # Update training options
    opts['checkpoint_path'] = PRETRAINED_WEIGHT
    opts['test_batch_size'] = 1
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded')

    predictor = dlib.shape_predictor(FACE_LANDMARKS_PATH)

    for i in range(num_images):
        start_time = time.time()
        res_path = os.path.join(RESULT_DATA_PATH, str(i))
        print(test_images_paths[i])
        test_image = Image.open(test_images_paths[i])
        aligned_image = align_face(image=test_image, predictor=predictor)
        aligned_image = aligned_image.convert('RGB')
        inp_img = INFERENCE_DATA_TRANSFORMS(aligned_image)
        if not os.path.exists(res_path):
            os.mkdir(res_path)

        with torch.no_grad():
            inference_time = time.time()
            enc_img, latents_dir = net(inp_img.to('cuda').float().unsqueeze(0), randomize_noise=False,
                                       return_latents=True)

            np_latents_dir = latents_dir.data.cpu().numpy()
            for i, dirs in enumerate(tqdm(latent_dirs)):
                dirs_res_path = os.path.join(res_path, str(latent_names[i]))
                if not os.path.exists(dirs_res_path):
                    os.mkdir(dirs_res_path)
                for coeff in COEFFS:
                    np_latents_dir[0][0:8] = (latents_dir.data.cpu().numpy()[0] + coeff * dirs)[0:8]

                    image, res_latents = net.decoder([torch.from_numpy(np_latents_dir).to('cuda').float()], 
                                                        input_is_latent=True, 
                                                        randomize_noise=False)
                    image = tensor2im(image[0])
                    image.save(os.path.join(dirs_res_path, '{}-coeffs-res.png'.format(coeff)))
        #np.save(os.path.join(RESULT_DATA_PATH, '{}-latent-dir.npy'.format(i)), latents_dir.data.cpu().numpy()[0])


if __name__ == "__main__":
    main()
