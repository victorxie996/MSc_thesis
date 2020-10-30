import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
from torch.nn.functional import interpolate
import torch
import torch.nn as nn
from math import log10
import torch.optim as optim
from utils import image_preprocessing
from model import _NetG
from SSIM import ssim
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

step = 500
device = torch.device("cuda")


def save_checkpoint(model, epoch):
    model_out_path = "../epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model, 'avg_psnr': avg_psnr_list}

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

device = torch.device("cuda")
model = _NetG().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

checkpoint = torch.load('epoch_60.pth')
model.load_state_dict(checkpoint['model'].state_dict())
avg_psnr_list = checkpoint['avg_psnr']


upscale_factor = 4
# change the file directory
image = nib.load('../Task07_Pancreas/imagesTr/pancreas_421.nii.gz')
image = np.asanyarray(image.dataobj)
image = image_preprocessing(image)
image = image.transpose((2,0,1))
input = torch.from_numpy(image).unsqueeze(1)
downsampled = interpolate(input, scale_factor = 1/upscale_factor,
                       mode='bilinear')
model_input = downsampled.to(device)
out = model(model_input)
out_img = out.cpu().detach().numpy()

bilinear_upsample = interpolate(downsampled, size = (512,512), mode = 'bilinear')
nearest_upsample = interpolate(downsampled, size = (512,512), mode = 'nearest')

depth = 10
def calculate_PSNR(input, prediction):
    criterion = nn.MSELoss()
    mse = criterion(input, prediction)
    psnr = 10 * log10(1/mse.item())
    return psnr

def calculate_SSIM(input, prediction):
    ssim_score = ssim(input, prediction)
    return ssim_score

x1, x2, y1, y2 = 250,300,350,400
locations = [x1, x2, y1, y2]
# locations_LR = x*0.5 for x in locations
def locations_LR(loc = locations):
  return [x*0.25 for x in loc]
locations_lr = locations_LR(locations)

fig, axs = plt.subplots(2,5,figsize=(20,10))
# LR
axs[0,0].imshow(downsampled[depth, 0, :, :], cmap='gray')
axs[0,0].set_title('low-resolution')

axs[1,0].imshow(downsampled[depth, 0, :, :], cmap='gray')
axs[1,0].set_xlim(locations_lr[0], locations_lr[1]) # apply the x-limits
axs[1,0].set_ylim(locations_lr[2], locations_lr[3]) # apply the y-limits

mark_inset(axs[0,0], axs[1,0], loc1=2, loc2=4, fc="none", ec="1")


# Original
axs[0,1].imshow(input[depth, 0, :, :], cmap='gray')
axs[0,1].set_title('Original')

axs[1,1].imshow(input[depth, 0, :, :], cmap='gray')
axs[1,1].set_xlim(locations[0], locations[1]) # apply the x-limits
axs[1,1].set_ylim(locations[2], locations[3]) # apply the y-limits
axs[1,1].set_title('PSNR/SSIM')
mark_inset(axs[0,1], axs[1,1], loc1=2, loc2=4, fc="none", ec="1")

# Binlinear
axs[0,2].imshow(bilinear_upsample[depth, 0, :, :], cmap='gray')
axs[0,2].set_title('Bilinear interpolation')

axs[1,2].imshow(bilinear_upsample[depth, 0, :, :], cmap='gray')
axs[1,2].set_xlim(locations[0], locations[1]) # apply the x-limits
axs[1,2].set_ylim(locations[2], locations[3]) # apply the y-limits
axs[1,2].set_title('{:.4f} dB/{:.4f} '.format(calculate_PSNR(bilinear_upsample, input),
                                                         calculate_SSIM(bilinear_upsample, input)))

mark_inset(axs[0,2], axs[1,2], loc1=2, loc2=4, fc="none", ec="1")

# Nearest

axs[0,3].imshow(nearest_upsample[depth, 0, :, :], cmap='gray')
axs[0,3].set_title('Nearest interpolation')

axs[1,3].imshow(nearest_upsample[depth, 0, :, :], cmap='gray')
axs[1,3].set_xlim(locations[0], locations[1]) # apply the x-limits
axs[1,3].set_ylim(locations[2], locations[3]) # apply the y-limits
axs[1,3].set_title('{:.4f} dB/{:.4f} '.format(calculate_PSNR(nearest_upsample, input),
                                                         calculate_SSIM(nearest_upsample, input)))
mark_inset(axs[0,3], axs[1,3], loc1=2, loc2=4, fc="none", ec="1")

# super-resolved

axs[0,4].imshow(out_img[depth, 0, :, :], cmap='gray')
axs[0,4].set_title('Super resolution')

axs[1,4].imshow(out_img[depth, 0, :, :], cmap='gray')
axs[1,4].set_xlim(locations[0], locations[1]) # apply the x-limits
axs[1,4].set_ylim(locations[2], locations[3]) # apply the y-limits
axs[1,4].set_title('{:.4f} dB/{:.4f} '.format(calculate_PSNR(out.cuda(), input.cuda()),
                                                         calculate_SSIM(out.cuda(), input.cuda())))
mark_inset(axs[0,4], axs[1,4], loc1=2, loc2=4, fc="none", ec="1")
