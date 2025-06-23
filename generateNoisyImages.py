import inspect

import os
# import cv2
#import matplotlib.pyplot as plt
# import mmcv
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import os.path as osp


def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


############################################################################################
####################### the code for low-light image generation part #######################
########(1)unprocess part(RGB2RAW) (2)low light corruption part (3)ISP part(RAW2RGB)########
############################################################################################
def get_mean_std(folder_path):
    """
    Calculate the mean and standard deviation of images in a folder.
    """
    mean = 0.0
    std = 0.0
    num_images = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.float32)
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
            num_images += 1

    mean /= num_images
    std /= num_images

    return mean, std
class Dark_ISP(object):
    '''
    darkness range : form
    '''

    def __init__(self,
                 img_path,
                 mean,
                 std,
                 darkness_range = (0.6, 1.0),
                 gamma_range=(1.0, 1.3),
                 rgb_range = (0.8, 1.0),
                 red_range = (1.0, 1.2),
                 blue_range = (1.0, 1.2),
                 quantization = [4, 6, 8],
                 train_mode = True
                 ):
        self.mean = mean
        self.std = std
        self.img_path = img_path
        self.darkness_low, self.darkness_high = darkness_range
        self.gamma_low, self.gamma_high = gamma_range
        #self.tone_curve = tone_curve
        #four camera type matrixs
        self.xyz2cams =[[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
        self.rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]
        self.rgb_mean, self.rgb_var = rgb_range
        self.red_low, self.red_high = red_range
        self.blue_low, self.blue_high = blue_range
        self.quantization = quantization
        self.train_mode = train_mode

    def __call__(self):
        """Call function to perform dark light noise distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images after dark_ISP process.
        """

        # if 'img_fields' in results:
        #     assert results['img_fields'] == ['img'], \
        #         'Only single img_fields is allowed'
        # img = results['img']
        img = Image.open(self.img_path).convert("RGB")
        img = np.array(img, dtype=np.float32)
        mean = self.mean
        std = self.std
        # mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        # std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = (img - mean) / std
        # img/=255.0
        #ori_img means the original normal light images before Dark_ISP process
        # ori_img = img.copy().astype(np.float32)
        # ori_filename = results['ori_filename'].replace('JPEGImages/','')
        
        #img_save_path = osp.join(save_path, ori_filename)
        #img_save_path1 = osp.join(save_path1, ori_filename)
        #img_save_path2 = osp.join(save_path2, ori_filename)
        #img_save_path3 = osp.join(save_path3, ori_filename)

        assert img.dtype == np.float32, \
            'This process needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        #print(img)
        #plt.imsave(img_save_path, img)
        img, meta_data = self.inverse_process(img)
        # print(meta_data)
        
        rgb_gain = 1.0/ meta_data['rgb_gain']
        red_gain = 1.0/ meta_data['red_gain']
        blue_gain = 1.0/ meta_data['blue_gain']
        gamma = 1.0/ meta_data['gamma']
        rgb2cam = meta_data['rgb2cam'] 
        #print('gamma', gamma, 'red_gain', red_gain, 'blue_gain', blue_gain, 'rgb2cam', rgb2cam, '\n')

        img, meta_data2 = self.low_light(img)
        k = meta_data2['k'] * rgb_gain
        quan = meta_data2['quan_noise']
        var = meta_data2['variance']
        # print(meta_data2)
        
        img = self.process(img,meta_data)
        img = np.clip(img , 0.0, 1.0)

        img = img.copy().astype(np.float32)
        return img


    # RGB2RAW(1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains)
    def inverse_process(self, img):
        gamma = np.random.uniform(self.gamma_low, self.gamma_high)
        #print('the gamma is :', gamma)
        img =  np.maximum(img, 1e-8) **gamma
        xyz2cam = random.choice(self.xyz2cams)
        rgb2xyz = np.array(self.rgb2xyz)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1)

        img = self.apply_ccm(img, rgb2cam)
        
        #4. inverse white balance and digital gain
        rgb_gain = 1.0 / np.random.normal(self.rgb_mean, self.rgb_var)
        red_gain = np.random.uniform(self.red_low, self.red_high)
        blue_gain = np.random.uniform(self.blue_low, self.blue_high)

        gains = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
        gains = gains[np.newaxis, np.newaxis, :]
        img =  img * gains
        
        meta_data = {'gamma': gamma, 
                    #'tone_type':tone_type, 
                    'rgb2cam':rgb2cam,
                    'red_gain':red_gain,
                    'blue_gain':blue_gain,
                    'rgb_gain':rgb_gain}
        #print(meta_data)
        return img, meta_data
    
    def low_light(self, img):
        #darkness
        darkness = np.random.uniform(self.darkness_low, self.darkness_high) 
        img = img * darkness
        print(darkness)
        #shot noise and read noise
        shot_noise, read_noise = self.random_noise_levels()
        variance = img * shot_noise + read_noise
        variance = np.maximum(variance, 1e-8)
        #print(variance)
        noise = np.random.normal(0, np.sqrt(variance), size = np.shape(img))
        img = img + noise
        #quantization step
        bits = random.choice(self.quantization)
        img_quan1 = img* 255.0
        img_quan_bit = img_quan1/ bits
        img_quan2 = np.around(img_quan_bit)* bits
        #quan_noise = (img_quan2 - img_quan1)/ bits
        img = img_quan2/ 255.0
        meta_data2 = {'k': darkness,
                      'quan_noise': 1/bits,
                      'variance': np.mean(variance)*1e3}
        return img, meta_data2
    
    # RAW2RGB(1.white balance 2.cam2rgb 3.gamma correction)
    def process(self, img, meta_data):
        gamma = meta_data['gamma']
        rgb2cam = meta_data['rgb2cam']
        red_gain = meta_data['red_gain']
        blue_gain = meta_data['blue_gain']

        #white balance
        green_gain = np.ones_like(red_gain)
        gains = np.stack([red_gain, green_gain, blue_gain], axis=-1)
        gains = gains[np.newaxis, np.newaxis, :]
        img = img * gains

        #cRGB2sRGB
        cam2rgb = np.linalg.inv(rgb2cam)
        img = self.apply_ccm(img, cam2rgb)

        #gamma correction
        img = np.maximum(img, 1e-8) **(1.0/ gamma)

        return img

    def apply_ccm(self, image, ccm):
        '''
        The function of apply CCM matrix
        '''
        shape = image.shape
        image = np.reshape(image, [-1, 3])
        image = np.tensordot(image, ccm, axes=[[-1], [-1]])
        return np.reshape(image, shape)
        
    def random_noise_levels(self):
        """Generates random noise levels from a log-log linear distribution."""
        log_min_shot_noise = np.log(0.0001)
        log_max_shot_noise = np.log(0.012)
        log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
        shot_noise = np.exp(log_shot_noise)

        line = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
        read_noise = np.exp(log_read_noise)
        return shot_noise, read_noise

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

num_lowlight_per_image = 1
output_path = "/mnt/disk1/data/environment_maps/low_light/relighting_market1501_x3_noise"
os.makedirs(output_path, exist_ok=True)
dataset_path = "/mnt/disk1/data/environment_maps/low_light/relighting_market1501_x3/"
mean, std = get_mean_std(dataset_path)
print(f"mean: {mean}, std: {std}")
img_list = [img for img in os.listdir(dataset_path) if img.endswith((".jpg", ".png"))]
for image in img_list:
    image_path = os.path.join(dataset_path, image)
    for i in range(num_lowlight_per_image):
        degrade_result = Dark_ISP(image_path, mean, std).__call__()
        degrade_result = (degrade_result * 255).astype(np.uint8)
        degrade_result_pil = Image.fromarray(degrade_result)
        save_path = output_path
        os.makedirs(save_path, exist_ok=True)
        # degrade_result_pil.save(os.path.join(save_path, f"{image.split('.')[0]}_{i}.jpg"))
        degrade_result_pil.save(os.path.join(save_path, f"{image.split('.')[0]}.png"))
    print(f"Finish processing {image_path}")
