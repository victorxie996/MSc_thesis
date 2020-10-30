import torch
import torch.nn as nn
from math import log10
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import _NetG
from SSIM import ssim
from datasets import get_training_set, get_test_set


step = 500
device = torch.device("cuda")

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = 1e-4 * (0.1 ** (epoch // step))
    return lr

def train(train_set, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    for iteration, batch in enumerate(train_set, 1):
        input, target = batch[0].cuda(), batch[1].cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 40 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(epoch, iteration, len(train_set), loss.item()))

def save_checkpoint(model, epoch):
    model_out_path = "../epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model,
             'avg_psnr': avg_psnr_list,
             'optimizer_state_dict': optimizer.state_dict(),
             'ssim_': avg_ssim_list}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

torch.cuda.manual_seed(12345)

cudnn.benchmark = True

print("===> Loading datasets")
train_set = get_training_set(batch_size = 16, upscale_factor = 4)
test_set = get_test_set(batch_size = 16, upscale_factor = 4)
print(len(train_set))
print(len(test_set))
print("===> Building model")
model = _NetG()
criterion = nn.MSELoss(size_average=False)

model = model.cuda()

print("===> Loading model")
weight = torch.load('../epoch_101.pth')
model.load_state_dict(weight['model'].state_dict())
avg_psnr_list = weight['avg_psnr']
avg_ssim_list = weight['ssim_']
criterion = nn.MSELoss()
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(weight['optimizer_state_dict'])

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
    print("===> epoch({}), Avg. PSNR: {:.6f} dB".format(epoch, avg_psnr / len(test_set)))

for epoch in range(1, 121):
    train(train_set, optimizer, model, criterion, epoch)
    test()
    save_checkpoint(model, epoch)
