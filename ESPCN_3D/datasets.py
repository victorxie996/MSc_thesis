import torch
import os
import nibabel as nib
import torch.utils.data as data
from utils import image_preprocessing
import numpy as np
from torch.nn.functional import interpolate


class TrainSetLoader(data.Dataset):
    def __init__(self, image_dir, batch_size, patch_length, upscale_factor):
        super(TrainSetLoader, self).__init__()
        self.image_name = image_dir
        self.batch_size = batch_size
        self.patch_length = patch_length
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        input = np.asanyarray(self.image_name[index].dataobj)
        input = image_preprocessing(input)
        input = input.transpose((2, 0, 1))
        if input.shape[0] <= self.patch_length:
            empty_slice_num = 64 - input.shape[0] + 1
            empty_slice = np.zeros((empty_slice_num, 512, 512))
            input = np.concatenate((input, empty_slice))

        patch_limit = int(self.patch_length / 2)
        batch_of_patch = np.zeros((self.batch_size, self.patch_length, self.patch_length, self.patch_length))

        image_shape = input.shape
        for i in range(self.batch_size):
            Width_Height = np.random.randint(patch_limit, image_shape[1] - patch_limit,
                                             size=2)  # get width, height location
            Depth = np.random.randint(patch_limit, image_shape[0] - patch_limit, size=1)  # get depth location
            D_W_H = [Depth[0], Width_Height[0], Width_Height[1]]  # together the location indices

            mini_patch = input[(D_W_H[0] - patch_limit):(D_W_H[0] + patch_limit),
                         (D_W_H[1] - patch_limit):(D_W_H[1] + patch_limit),
                         (D_W_H[2] - patch_limit):(D_W_H[2] + patch_limit)]

            batch_of_patch[i, :, :, :] = mini_patch

        input = batch_of_patch[:, np.newaxis, :, :, :]
        input = torch.from_numpy(input).float()
        target = input.clone()

        input = interpolate(input,
                            scale_factor=1 / self.upscale_factor,
                            mode='trilinear')

        return input, target


class TestSetLoader(data.Dataset):
    def __init__(self, image_dir, upscale_factor):
        super(TestSetLoader, self).__init__()
        self.image_name = image_dir
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        input = np.asanyarray(self.image_name[index].dataobj)
        input = image_preprocessing(input)
        input = input.transpose((2, 0, 1))

        input_shape = input.shape  # now is [depth, Height, Width]

        if input_shape[0] % self.upscale_factor != 0:
            remainder = self.upscale_factor - (input_shape[0] % self.upscale_factor)
            remainder_slice = np.zeros(
                (remainder, input_shape[1], input_shape[2]))  # create zeros matrices as nX512X512
            input = np.concatenate((input, remainder_slice))  # add empty slices at bottom

        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
        target = input.clone()

        input = interpolate(input,
                            scale_factor=1 / self.upscale_factor,
                            mode='trilinear')

        return input.float(), target.float()

# load the abdominal dataset. the full dataset can be download from
# http://medicaldecathlon.com/.
# Once the dataset is downloaded, change the directory name of imagesTr and imagesTs

imagesTr = ['../Task07_Pancreas/imagesTr'] # load the image Tr dataset
imagesTr_dir = []
for folder_imagesTr in imagesTr:
  for _, _, files in os.walk(folder_imagesTr):
      for file in files:
        if not file[0] == '.':
          imagesTr_dir.append(os.path.join(folder_imagesTr, file))

imagesTs = ['../Task07_Pancreas/imagesTs'] # load the image Ts dataset
imagesTs_dir = []
for folder_imagesTs in imagesTs:
  for _, _, files_imagesTs in os.walk(folder_imagesTs):
      for file in files_imagesTs:
        if not file[0] == '.':
          imagesTs_dir.append(os.path.join(folder_imagesTs, file))

def load_pancreas_3D():
    imagesTr_list = []

    for index, image in enumerate(imagesTr_dir):
        imagesTr_list.append(nib.load(image))
        if index%100 == 0:
            print('imagesTr:{}'.format(index))

    imagesTs_list = []
    for index, image in enumerate(imagesTs_dir):
        imagesTs_list.append(nib.load(image))
        if index%100 == 0:
            print('imagesTs:{}'.format(index))

    return imagesTr_list, imagesTs_list


train_test_split_Tr = 251

def get_training_set(batch_size, patch_length, upscale_factor):
    train_dir = load_pancreas_3D()

    train_Tr = TrainSetLoader(image_dir = train_dir[0][:train_test_split_Tr],
                                 batch_size = batch_size,
                                 patch_length = patch_length,
                                 upscale_factor = upscale_factor)
    train_Ts = TrainSetLoader(image_dir = train_dir[1],
                                 batch_size = batch_size,
                                 patch_length = patch_length,
                                 upscale_factor = upscale_factor)
    return train_Tr + train_Ts

def get_test_set(upscale_factor):
    test_dir = load_pancreas_3D()
    test_Tr = TestSetLoader(image_dir = test_dir[0][train_test_split_Tr:],
                               upscale_factor = upscale_factor)
    return test_Tr