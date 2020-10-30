from datasets import get_training_set, get_test_set
from SSIM import ssim
from model import Net
import torch.optim as optim
import torch
from math import log10
import torch.nn as nn
import time

train_set = get_training_set(batch_size = 32, patch_length= 64, upscale_factor = 4)
test_set = get_test_set(upscale_factor = 4)

device = torch.device("cuda")
model = Net(upscale_factor = 4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# if use pretrained data:
# checkpoint = torch.load('../iteration_60.pth')
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# model.load_state_dict(checkpoint['model'])
# loss = checkpoint['loss']
# avg_psnr_list = checkpoint['avg_psnr']
# avg_ssim_list = checkpoint['ssim_']

avg_psnr_list = []
loss_list = []
avg_ssim_list = []

def train(iteration):
    iteration_loss = 0
    for iter, batch in enumerate(train_set, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        iteration_loss += loss.item()
        loss.backward()
        optimizer.step()
        loss_list.append(iteration_loss)
        if iter %40 == 0:
          print("===> Iteration[{}]({}/{}): Loss: {:.6f}".format(iteration, iter, len(train_set), loss.item()))

    print("===> Iteration {} Complete: Avg. Loss: {:.6f}".format(iteration, iteration_loss / len(train_set)))

def test():
    avg_psnr = 0
    avg_ssim_score = 0
    with torch.no_grad():
        for batch in test_set:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            ssim_score = ssim(prediction, target)
            avg_ssim_score += ssim_score
        avg_psnr_list.append(avg_psnr / len(test_set))
        avg_ssim_list.append(avg_ssim_score/len(test_set))
    print("===> Iteration({}), Avg. PSNR: {:.6f} dB".format(iteration, avg_psnr / len(test_set)))
            #ssim calculation
    print("===> Iteration({}), SSIM score: {:.6f} ".format(iteration, avg_ssim_score/len(test_set)))

def checkpoint(iteration):
    model_out_path = "../uf=4_iteration_{}.pth".format(iteration)

    torch.save({'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'avg_psnr': avg_psnr_list,
                'ssim_': avg_ssim_list
                }, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

start_time = time.time()

for iteration in range(1,61):
    train(iteration)
    test()
    checkpoint(iteration)

print('End of 20 epoches \t Time Taken: %d sec' % (time.time() - start_time))