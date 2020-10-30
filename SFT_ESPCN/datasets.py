import torch.utils.data as data
import torch
import numpy as np
from torch.nn.functional import interpolate
import nibabel as nib
import os

class TrainSetLoader_Tr(data.Dataset):
    def __init__(self, imagesTr, labelsTr, batch_size, upscale_factor):
        super(TrainSetLoader_Tr, self).__init__()
        self.imagesTr = imagesTr
        self.labelsTr = labelsTr
        self.batch_size = batch_size
        self.upscale_factor = upscale_factor
    def __len__(self):
        return len(self.imagesTr)

    def __getitem__(self, index):
      # get input Tr data
      input_Tr_img = np.asanyarray(self.imagesTr[index].dataobj)
      depth = input_Tr_img.shape[2]
      ma, mi = np.max(input_Tr_img), np.min(input_Tr_img) # normalization
      input_Tr_img = (input_Tr_img - mi) / (ma - mi)
      input_Tr_img = input_Tr_img.transpose(2, 0, 1)
      new_depth_range = np.random.choice(range(depth), self.batch_size, replace = False)
      input_Tr_img = input_Tr_img[new_depth_range, :, :] # select batches
      input_Tr_img = torch.from_numpy(input_Tr_img).unsqueeze(1).float()
      target = input_Tr_img.clone()
      input_Tr_img = interpolate(input_Tr_img, scale_factor = 1/self.upscale_factor,
                                  mode = 'bilinear', align_corners = False)
      # get segementation probability maps
      input_Tr_lab = np.asanyarray(self.labelsTr[index].dataobj)
      seg_prob_maps = np.zeros((depth, 2, 512, 512))
      seg_pancreas = np.where(input_Tr_lab < 2, input_Tr_lab, 1)
      seg_tumor = np.where(input_Tr_lab > 1, input_Tr_lab, 0)
      seg_tumor = np.where(seg_tumor != 2, seg_tumor, 1)
      seg_prob_maps[:, 0, :, :] = seg_pancreas.transpose(2, 0, 1) # pancreas channel
      seg_prob_maps[:, 1, :, :] = seg_tumor.transpose(2, 0, 1) # tumors channel
      seg_prob_maps = seg_prob_maps[new_depth_range, :, :, :] # select batches
      seg_prob_maps = torch.from_numpy(seg_prob_maps).float()
      input = (input_Tr_img, seg_prob_maps)
      return input, target


class TestSetLoader_Tr(data.Dataset):
    def __init__(self, imagesTr, labelsTr, batch_size, upscale_factor):
        super(TestSetLoader_Tr, self).__init__()
        self.imagesTr = imagesTr
        self.labelsTr = labelsTr
        self.batch_size = batch_size
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.imagesTr)

    def __getitem__(self, index):
        input_Tr_img = np.asanyarray(self.imagesTr[index].dataobj)
        depth = input_Tr_img.shape[2]
        ma, mi = np.max(input_Tr_img), np.min(input_Tr_img)  # normalization
        input_Tr_img = (input_Tr_img - mi) / (ma - mi)
        input_Tr_img = input_Tr_img.transpose(2, 0, 1)
        new_depth_range = np.random.choice(range(depth), self.batch_size, replace=False)
        input_Tr_img = input_Tr_img[new_depth_range, :, :]  # select batches
        input_Tr_img = torch.from_numpy(input_Tr_img).unsqueeze(1).float()
        target = input_Tr_img.clone()
        input_Tr_img = interpolate(input_Tr_img, scale_factor=1 / self.upscale_factor,
                                   mode='bilinear', align_corners=False)
        # get segementation probability maps
        input_Tr_lab = np.asanyarray(self.labelsTr[index].dataobj)
        seg_prob_maps = np.zeros((depth, 2, 512, 512))
        seg_pancreas = np.where(input_Tr_lab < 2, input_Tr_lab, 1)
        seg_tumor = np.where(input_Tr_lab > 1, input_Tr_lab, 0)
        seg_tumor = np.where(seg_tumor != 2, seg_tumor, 1)
        seg_prob_maps[:, 0, :, :] = seg_pancreas.transpose(2, 0, 1)  # pancreas channel
        seg_prob_maps[:, 1, :, :] = seg_tumor.transpose(2, 0, 1)  # tumors channel
        seg_prob_maps = seg_prob_maps[new_depth_range, :, :, :]  # select batches
        seg_prob_maps = torch.from_numpy(seg_prob_maps).float()
        input = (input_Tr_img, seg_prob_maps)
        return input, target

# change the directory path
imagesTr = ['../Task07_Pancreas/imagesTr']
imagesTr_dir = []
for folder_imagesTr in imagesTr:
  for _, _, files in os.walk(folder_imagesTr):
      for file in files:
        if not file[0] == '.':
          imagesTr_dir.append(os.path.join(folder_imagesTr, file))

path = '../Task07_Pancreas/labelsTr'
labelsTr_files = os.listdir(path)
labelsTr_files.sort()

labelsTr_dir = []

for file_ in labelsTr_files:
  labelsTr_dir.append(os.path.join(path, file_))

def load_pancreas_2D_seg():
  imagesTr_list = []

  for index, image in enumerate(imagesTr_dir):
    imagesTr_list.append(nib.load(image))
    if index%100 == 0:
      print('imagesTr:{}'.format(index))

  labelsTr_list = []

  for index, image in enumerate(labelsTr_dir):
    labelsTr_list.append(nib.load(image))
    if index%100 == 0:
      print('labelsTr:{}'.format(index))

  return imagesTr_list, labelsTr_list

train_test_split = 250
def get_training_set_seg(batch_size, upscale_factor):
    train_dir = load_pancreas_2D_seg()
    train_Tr = TrainSetLoader_Tr(imagesTr = train_dir[0][:train_test_split],
                                labelsTr = train_dir[1][:train_test_split],
                                batch_size = batch_size,
                                upscale_factor = upscale_factor)
    return train_Tr

def get_test_set_seg(batch_size, upscale_factor):
    test_dir = load_pancreas_2D_seg()
    test_Tr = TestSetLoader_Tr(imagesTr = test_dir[0][train_test_split:],
                               labelsTr = test_dir[1][train_test_split:],
                               batch_size = batch_size,
                               upscale_factor = upscale_factor)
    return test_Tr
