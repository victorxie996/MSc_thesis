import torch.nn as nn
import torch.nn.init as init

class pixel_shuffle_3D(object):
  def __init__(self, upscale_factor):
    self.upscale_factor = upscale_factor

  def __call__(self, input):

    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (self.upscale_factor ** dimensionality)
    output_size = [dim * self.upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([self.upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))

    shuffle_out = input_view
    return shuffle_out.permute(0, 1, 5, 2,
                               6, 4, 7, 3).contiguous().view(input_size[0], input_size[1], *output_size)


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, 64, 5, 1, 2)
        self.conv2 = nn.Conv3d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv3d(64, 32, 3,1, 1)
        self.conv4 = nn.Conv3d(32, upscale_factor ** 3, 3, 1, 1)
        self.pixel_shuffle = pixel_shuffle_3D(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
