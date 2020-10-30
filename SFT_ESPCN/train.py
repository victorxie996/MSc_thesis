import torch
import torch.nn as nn
from math import log10
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import SFT_Net
from SSIM import ssim
from datasets import get_training_set_seg, get_test_set_seg

train_set = get_training_set_seg(batch_size = 16, upscale_factor = 4)
test_set = get_test_set_seg(batch_size = 16, upscale_factor = 4)

device = torch.device("cuda")
model = SFT_Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

checkpoint = torch.load('../iteration_120.pth')
model.load_state_dict(checkpoint['model'])
loss = checkpoint['loss']
avg_psnr = checkpoint['avg_psnr']

criterion = nn.MSELoss()

def train(iteration):
  iteration_loss = 0
  for iteration, batch in enumerate(train_set, 1):
        input, target = batch[0], batch[1].to(device)
        optimizer.zero_grad()
        model_output = model((input[0].to(device), input[1].to(device)))

        loss = criterion(model_output, target)
        iteration_loss += loss.item()
        loss.backward()
        optimizer.step()
        loss_list.append(iteration_loss)
        if iteration %40 == 0:
          print("===> Iteration[{}]({}/{}): Loss: {:.6f}".format(iteration, iteration, len(train_set), loss.item()))

  print("===> Iteration {} Complete: Avg. Loss: {:.6f}".format(iteration, iteration_loss / len(train_set)))

def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in test_set:
            input, target = batch[0], batch[1].to(device)

            prediction = model((input[0].to(device), input[1].to(device)))
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    avg_psnr_list.append(avg_psnr / len(test_set))
    print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(test_set)))

avg_psnr_list = []
loss_list = []

def checkpoint(iteration):
    model_out_path = "../iteration_{}.pth".format(iteration)

    torch.save({#'iteration': iteration,
                'model': model.state_dict(),
                'model_only': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'avg_psnr': avg_psnr_list,
                }, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

for iteration in range(74, 121):
    train(iteration)
    test()
    checkpoint(iteration)

