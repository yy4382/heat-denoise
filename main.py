import os
import cv2
import numpy as np
from denoise import *


def get_images(num=None) -> list:
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


def generate_image(
    images: list, noisy_images: list, denoised_images: list, params: DenoiseParams
) -> np.ndarray:
    imgs = np.hstack((np.vstack(images), np.vstack(noisy_images), np.vstack(denoised_images)))
    width = imgs.shape[1]
    header = np.zeros((50, width), dtype=np.uint8)
    cv2.putText(header, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(
        header, "Noisy", (10 + width // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.putText(
        header,
        "Denoised",
        (10 + width * 2 // 3, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    footer = np.zeros((50, width), dtype=np.uint8)
    cv2.putText(
        footer,
        str(params),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    imgs = np.vstack((header, imgs, footer))
    print(imgs.dtype)
    return imgs


if __name__ == "__main__":
    # metadata
    params = DenoiseParams(
        d_type=DenoiseType.ANALYTIC, a=1, t=1, MAX=5000, lam=30, target_size=256
    )

    image_files = get_images()
    images = [image2matrix(image, params.target_size) for image in image_files]
    noisy_images = [add_noise(image, params.lam) for image in images]

    denoised_images = [denoise_img(noisy_image, params) for noisy_image in noisy_images]

    result_img = generate_image(images, noisy_images, denoised_images, params)
    cv2.imwrite(f"data/results/{params}.png", result_img)
    cv2.imshow("imgs", result_img)
    cv2.waitKey(0)
