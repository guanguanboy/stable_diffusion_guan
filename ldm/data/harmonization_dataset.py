import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


import torchvision.transforms.functional as tf

class HarmonizationBase(data.Dataset):
    def __init__(self, data_root, sub_dir, size=256):
        imgs = make_dataset(data_root)

        self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = pil_loader
        self.image_size = size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.sub_dir = sub_dir

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        name_parts=path.split('_')
        mask_path = self.imgs[index].replace(self.sub_dir,'masks') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.imgs[index].replace(self.sub_dir,'real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[self.image_size, self.image_size])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)

        ret['gt_image'] = real
        ret['cond_image'] = comp
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]

        return ret

    def __len__(self):
        return len(self.imgs)

class HarmonizationTrain(HarmonizationBase):
    def __init__(self, **kwargs):
        #super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_train_without_noise/", sub_dir='composite_images_train_without_noise', **kwargs)
        super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/HAdobe5k/composite_images_train/", sub_dir='composite_images_train', **kwargs)

class HarmonizationValidation(HarmonizationBase):
    def __init__(self, **kwargs):
        #super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test_without_noise/", sub_dir='composite_images_test_without_noise', **kwargs)
        super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test/", sub_dir='composite_images_test', **kwargs)


class HarmonizationDay2nightTrain(HarmonizationBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_train_without_noise/", sub_dir='composite_images_train_without_noise', **kwargs)

class HarmonizationDay2nightValidation(HarmonizationBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/data1/liguanlin/Datasets/iHarmony/Hday2night/composite_images_test_without_noise/", sub_dir='composite_images_test_without_noise', **kwargs)

class SSHarmonizationTestDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        # data_root的样例../RealHM
        self.image_small_list = []
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_1/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_2/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
                    
        for root, dirs, files in os.walk(data_root  + "/vendor_testing_3/"):
            for file in files:
                if "small" in file:
                    self.image_small_list.append(os.path.join(root, file))
        
        
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

    def __getitem__(self, index):
        ret = {}
        path = self.image_small_list[index]

        comp_path = path.replace("_small.jpg", "_composite_noise25.jpg")
        mask_path = path.replace("_small.jpg", "_mask.jpg")
        target_path = path.replace("_small.jpg", "_gt.jpg")

        comp = Image.open(comp_path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [self.image_size[0], self.image_size[1]])
        mask = tf.resize(mask, [self.image_size[0], self.image_size[1]])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[self.image_size[0],self.image_size[1]])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)

        ret['gt_image'] = real
        ret['cond_image'] = comp
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]

        return ret

    def __len__(self):
        return len(self.image_small_list)


