import numpy as np
from skimage.io import imsave
from skimage.color import rgb2gray
from cv2 import GaussianBlur, imread

def line_extraction(image, gamma, sigma, k, epsilon, phi):

    if image.shape[2] == 3:
        image = rgb2gray(image)

    image1 = GaussianBlur(image, (0,0), sigma)
    image2 = GaussianBlur(image, (0,0), sigma * k)

    difference = image1 - gamma*image2

    mask = difference < epsilon

    difference[mask] = 1
    difference[~mask] = 1 + np.tanh(phi * difference[~mask])

    return difference

if __name__ == '__main__':

    name = 'Test.jpg'
    image = imread(name)
    line = line_extraction(image, gamma = 0.97, sigma = 1, k = 1.2, epsilon = -0.1, phi = 200)

    imsave('output.jpg', line)