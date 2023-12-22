import os
import cv2
import numpy as np
from denoise import *


def get_images(path="data/original", num=None) -> list:
    images = [path + "/" + image for image in os.listdir(path)]
    if num is not None:
        return images[:num]
    return images


def image2matrix(image: str, target_size) -> np.ndarray:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (target_size, target_size))
    return img


def add_noise(image: np.ndarray, params: DenoiseParams) -> np.ndarray:
    def gaussian_noise(img: np.ndarray, mean, sigma):
        img = img / 255
        noise = np.random.normal(mean, sigma, img.shape)
        gaussian_out: np.ndarray = img + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = (gaussian_out * 255).astype(np.uint8)
        return gaussian_out

    def poisson_noise(img, lam):
        noisy_type = np.random.poisson(lam=lam, size=img.shape).astype(dtype="uint8")
        noisy_image = noisy_type + img
        return noisy_image

    if params.noise_type == NoiseType.POISSON:
        return poisson_noise(image, params.poisson_lam)
    elif params.noise_type == NoiseType.GAUSS:
        return gaussian_noise(image, params.gaussian_mean, params.gaussian_sigma)
    else:
        raise ValueError("Unknown noise type")


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
        0.6,
        (255, 255, 255),
        2,
    )
    imgs = np.vstack((header, imgs, footer))
    return imgs

def default_main():
    """
    默认的main函数，生成带标签的图片
    """
    # metadata
    params = DenoiseParams(
        d_type=DenoiseType.ITERATIVE,
        a=1 / 40,
        t=100000,
        MAX=20000,
        lam=35,
        target_size=256,
        noise_type=NoiseType.GAUSS,
        gaussian_mean=0,
        gaussian_sigma=0.03,
        poisson_lam=35,
    )
    image_files = get_images(num=1)
    images = [image2matrix(image, params.target_size) for image in image_files]
    noisy_images = [add_noise(image, params) for image in images]

    denoised_images = [denoise_img(noisy_image, params) for noisy_image in noisy_images]
    result_img = generate_image(images, noisy_images, denoised_images, params)
    cv2.imwrite(f"data/results/{params}.png", result_img)
    return result_img

def compare_main():
    """
    用来搞一些自定义对比
    """
    paramsList = []
    for tt in range(6):
        paramsList.append(
            DenoiseParams(
                d_type=DenoiseType.ANALYTIC,
                a=1,
                t=tt/10,
                MAX=10000,
                lam=35,
                target_size=256,
                noise_type=NoiseType.GAUSS,
                gaussian_mean=0,
                gaussian_sigma=0.03,
                poisson_lam=35,
            )
        )
    image_files = get_images(num=1)
    images = [image2matrix(image, paramsList[0].target_size) for image in image_files]
    noisy_images = [add_noise(image, paramsList[0]) for image in images]
    denoised_images = [denoise_img(noisy_images[0], params1) for params1 in paramsList]
    result_img = np.hstack(denoised_images)
    return result_img

if __name__ == "__main__":
    result_img = compare_main()
    # result_img = default_main()
    cv2.imshow("imgs", result_img)
    cv2.waitKey(0)
