'''
0   -   Stylization(50, 0.6)
1   -   Stylization(50, 0.3)
2   -   Stylization(50, 0.1)
3   -   Gaussian noise(0.2)
4   -   Gaussian noise(0.4)
5   -   Gaussian noise(0.8)
6   -   Crop(4)
7   -   Crop(3)
8   -   Crop(2)
9   -   Quantization(64)
10  -   Quantization(32)
11  -   Quantization(8)
12  -   Gaussian blur(0.4)
13  -   Gaussian blur(0.8)
14  -   Gaussian blur(2)
15  -   Convex(1/8)
16  -   Convex(1/4)
17  -   Convex(1/2)
18  -   Pencil Sketch(100, 0.1, 0.02)
19  -   Pencil Sketch(100, 0.4, 0.02)
20  -   Pencil Sketch(100, 0.6, 0.02)
21  -   exposure(1.5)
22  -   exposure(2.0)
23  -   exposure(2.5)
24  -   CutMix(32)
25  -   CutMix(64)
26  -   CutMix(128)
27  -   Rotation(45)
28  -   Rotation(-45)
29  -   None
'''

import cv2 as cv
import numpy as np
import random
import math


def detail_enhance(image, sigma_s, sigma_r):
    degraded_img = cv.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)

    return degraded_img

def stylization(image, sigma_s, sigma_r):
    degraded_img = cv.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)

    return degraded_img

def pencil_sketch(image, sigma_s, sigma_r, shade_factor):
    degraded_img = cv.pencilSketch(image, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)

    return degraded_img

# Add gaussian noise
def add_gaussian_noise(image, sigma):
    image = image / 255
    noise = np.random.normal(0, sigma, image.shape)
    degraded_img = image + noise
    degraded_img = np.clip(degraded_img, 0, 1)

    degraded_img = degraded_img * 255
    degraded_img = np.uint8(degraded_img)

    return degraded_img

# color quantization (according to https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/ch48-K%E5%80%BC%E8%81%9A%E7%B1%BB/48.2.3_%E9%A2%9C%E8%89%B2%E9%87%8F%E5%8C%96.py)
def quantization(image, K):
    tmp = image.reshape((-1, 3))
    tmp = np.float32(tmp)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv.kmeans(
        tmp, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    degraded_img = center[label.flatten()]
    degraded_img = degraded_img.reshape((image.shape))

    return degraded_img

# Gaussian blur
def gaussian_blur(image, std):
    degraded_img = cv.GaussianBlur(image, (5, 5), std)

    return degraded_img

# crop and resize
def crop_resize(image, factor):
    h, w, c = image.shape
    degraded_img = image[0:h - h // factor, 0:w - w // factor, :]
    degraded_img = cv.resize(degraded_img, (w, h))

    return degraded_img

# convex lens
def convex(image, factor):
    h, w, c = image.shape
    center_x, center_y = h/2, w/2
    radius = math.sqrt(h**2+w**2)/factor
    degraded_img = image.copy()
    for i in range(h):
        for j in range(w):
            dis = math.sqrt((i-center_x)**2+(j-center_y)**2)
            if dis <= radius:
                new_i = int(np.round(dis/radius*(i-center_x)+center_x))
                new_j = int(np.round(dis/radius*(j-center_y)+center_y))
                degraded_img[i, j] = image[new_i, new_j]
    return degraded_img

# change exposure
def exposure(image, factor):
    degraded_img = image * factor
    degraded_img = np.clip(degraded_img, 0, 255)
    degraded_img = np.uint8(degraded_img)

    return degraded_img

# rotate
def rotate(image, deg):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    angle = deg
    scale = 0.8

    M = cv.getRotationMatrix2D(center, angle, scale)
    image_rotation = cv.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(255, 255, 255))

    return image_rotation

# cutmix
def cutmix(image, image_mixup, size):
    a = random.randint(0, 224 - size - 1)
    b = random.randint(0, 224 - size - 1)

    degraded_img = image.copy()
    degraded_img[a:a + size, b:b + size, :] = image_mixup[a:a + size, b:b + size, :]
    return degraded_img

# Choose manipulations
def image_manipulation(image, opt, image_mixup):
    if opt == 0:
        degraded_image = stylization(image, 50, 0.6)
    elif opt == 1:
        degraded_image = stylization(image, 50, 0.3)
    elif opt == 2:
        degraded_image = stylization(image, 50, 0.1)
    elif opt == 3:
        degraded_image = add_gaussian_noise(image, 0.2)
    elif opt == 4:
        degraded_image = add_gaussian_noise(image, 0.4)
    elif opt == 5:
        degraded_image = add_gaussian_noise(image, 0.8)
    elif opt == 6:
        degraded_image = crop_resize(image, 4)
    elif opt == 7:
        degraded_image = crop_resize(image, 3)
    elif opt == 8:
        degraded_image = crop_resize(image, 2)
    elif opt == 9:
        degraded_image = quantization(image, 64)
    elif opt == 10:
        degraded_image = quantization(image, 32)
    elif opt == 11:
        degraded_image = quantization(image, 8)
    elif opt == 12:
        degraded_image = gaussian_blur(image, 0.4)
    elif opt == 13:
        degraded_image = gaussian_blur(image, 0.8)
    elif opt == 14:
        degraded_image = gaussian_blur(image, 2)
    elif opt == 15:
        degraded_image = convex(image, 8)
    elif opt == 16:
        degraded_image = convex(image, 4)
    elif opt == 17:
        degraded_image = convex(image, 2)
    elif opt == 18:
        _, degraded_image = pencil_sketch(image, 100, 0.1, 0.02)
    elif opt == 19:
        _, degraded_image = pencil_sketch(image, 100, 0.4, 0.02)
    elif opt == 20:
        _, degraded_image = pencil_sketch(image, 100, 0.6, 0.02)
    elif opt == 21:
        degraded_image = exposure(image, 1.5)
    elif opt == 22:
        degraded_image = exposure(image, 2.0)
    elif opt == 23:
        degraded_image = exposure(image, 2.5)
    elif opt == 24:
        degraded_image = cutmix(image, image_mixup, 32)
    elif opt == 25:
        degraded_image = cutmix(image, image_mixup, 64)
    elif opt == 26:
        degraded_image = cutmix(image, image_mixup, 128)
    elif opt == 27:
        degraded_image = rotate(image, 45)
    elif opt == 28:
        degraded_image = rotate(image, -45)
    elif opt == 29:
        degraded_image = image.copy()

    return degraded_image
