import os
import cv2
import numpy as np
from denoise import *

def get_images(num = None) -> list:
    path = "data/original"
    images = [path + "/" + image for image in os.listdir(path)]
    if num is not None:
        return images[:num]
    return images


def image2matrix(image: str, target_size) -> np.ndarray:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (target_size, target_size))
    return img


def add_noise(image: np.ndarray, lam: int) -> np.ndarray:
    noisy_type = np.random.poisson(lam=lam, size=image.shape).astype(dtype="uint8")
    noisy_image = noisy_type + image
    return noisy_image


if __name__ == "__main__":
    # metadata
    d_type = DenoiseType.ANALYTIC
    target_size = 256
    a = 1
    t = 1
    MAX = 5000
    lam = 0

    image_files = get_images(1)
    images = [image2matrix(image, target_size) for image in image_files]
    noisy_images = [add_noise(image, lam) for image in images]

    denoised_images = [denoise_img(noisy_image, d_type, a, t, MAX) for noisy_image in noisy_images]

    imgs = np.hstack((np.vstack(images), np.vstack(noisy_images), np.vstack(denoised_images)))
    cv2.imwrite(f"data/results/{d_type.name}_a{a}_t{t}_MAX{MAX}_lam{lam}.png", imgs)
    cv2.imshow("imgs", imgs)
    cv2.waitKey(0)
