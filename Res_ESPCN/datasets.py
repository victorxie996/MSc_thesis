import torch.utils.data as data
import torch
import numpy as np
from torch.nn.functional import interpolate
import nibabel as nib
import os

class TrainSetLoader(data.Dataset):
    def __init__(self, image_dir, batch_size, upscale_factor):
        super(TrainSetLoader, self).__init__()
        self.image_name = image_dir
        self.batch_size = batch_size
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        input = np.asanyarray(self.image_name[index].dataobj)
        input = image_preprocessing(input)
        depth_range = np.random.randint(0, input.shape[2], size=self.batch_size)
        input = input[:, :, depth_range]
        input = input.transpose((2, 0, 1))[:, np.newaxis, :, :]
        input = torch.from_numpy(input).float()
        target = input.clone()

        input = interpolate(input, scale_factor=1 / self.upscale_factor,
                            mode='bilinear')
        return input, target

class TestSetLoader(data.Dataset):
    def __init__(self, image_dir, batch_size, upscale_factor):
        super(TestSetLoader, self).__init__()
        self.image_name = image_dir
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        input = np.asanyarray(self.image_name[index].dataobj)  # self.image_name[index].get_data()
        input = image_preprocessing(input)

        input = input.transpose((2, 0, 1))[:, np.newaxis, :, :]
        new_depth_range = np.random.choice(range(input.shape[0]), self.batch_size, replace=False)
        input = input[new_depth_range, :, :]

        input = torch.from_numpy(input).float()
        target = input.clone()

        input = interpolate(input,
                            scale_factor=1 / self.upscale_factor,
                            mode='bilinear')

        return input, target

# load the abdominal dataset. the full dataset can be download from
# http://medicaldecathlon.com/.
# Once the dataset is downloaded, change the directory name of imagesTr and imagesTs

imagesTr = ['../Task07_Pancreas/imagesTr']
imagesTr_dir = []
for folder_imagesTr in imagesTr:
  for _, _, files in os.walk(folder_imagesTr):
      for file in files:
        if not file[0] == '.':
          imagesTr_dir.append(os.path.join(folder_imagesTr, file))

imagesTs = ['../Task07_Pancreas/imagesTs']
imagesTs_dir = []
for folder_imagesTs in imagesTs:
  for _, _, files_imagesTs in os.walk(folder_imagesTs):
      for file in files_imagesTs:
        if not file[0] == '.':
          imagesTs_dir.append(os.path.join(folder_imagesTs, file))
def load_pancreas_2D():
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
def get_training_set(batch_size, upscale_factor):
    train_dir = load_pancreas_2D()

    train_Tr = TrainSetLoader(image_dir = train_dir[0][:train_test_split_Tr],
                                 batch_size = batch_size,
                                 upscale_factor = upscale_factor)
    train_Ts = TrainSetLoader(image_dir = train_dir[1],
                                 batch_size = batch_size,
                                 upscale_factor = upscale_factor)
    return train_Tr + train_Ts

def get_test_set(batch_size, upscale_factor):
    test_dir = load_pancreas_2D()
    test_Tr = TestSetLoader(image_dir = test_dir[0][train_test_split_Tr:],
                                batch_size = batch_size,
                               upscale_factor = upscale_factor)
    return test_Tr