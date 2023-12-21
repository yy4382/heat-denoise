import os
import cv2
import numpy as np
import math


def get_images() -> list:
    path = "data/original"
    images = [path + "/" + image for image in os.listdir(path)]
    return images


def image2matrix(image: str) -> np.ndarray:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    return img


def denoise(image: np.ndarray, *, t: float, a: float) -> np.ndarray:
    MAX = 5
    height = image.shape[1]
    width = image.shape[0]

    def cal_phi(nx, ny) -> float:
        result = 0.0
        for i in range(width):
            for j in range(height):
                result += (
                    image[j][i]
                    * math.cos(math.pi * ny * i / width)
                    * math.cos(math.pi * nx * j / height)
                )
        result = result * 4 / width / height
        return result

    def cal_miu(nx, ny) -> float:
        return (math.pi * nx / height) ** 2 + (math.pi * ny / width) ** 2

    def cal_dot(x, y) -> int:
        result = 0.0
        for nx in range(1, MAX):
            for ny in range(1, MAX):
                print(cal_phi(nx, ny))
                result += (
                    cal_phi(nx, ny)
                    * math.exp(-a * cal_miu(nx, ny) * t)
                    * math.cos(math.pi * nx * x / height)
                    * math.cos(math.pi * ny * y / width)
                )
        return max(0, int(result))

    denoised_image = np.zeros((width, height), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            dot = cal_dot(i, j)
            # assert dot < 0
            denoised_image[i][j] = dot
            # print(i, j, dot)
    return denoised_image


if __name__ == "__main__":
    images = get_images()
    # cv2.imshow("original_image", image2matrix(images[0]))
    # cv2.waitKey(0)
    denoised_image = denoise(image2matrix(images[0]), a=0.1, t=0.1)
    cv2.imshow("denoised_image", denoised_image)
    cv2.waitKey(0)