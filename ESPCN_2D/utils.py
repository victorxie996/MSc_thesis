import numpy as np
from scipy.ndimage import gaussian_filter

def image_preprocessing(image, sigma = 1, kernel_size = 3):
    ma, mi = np.max(image), np.min(image)
    image = (image - mi) / (ma - mi)

    truncate = (((kernel_size - 1)/2)-0.5)/sigma
    image = gaussian_filter(image, sigma = sigma, truncate= truncate)
    return image


