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
    img = cv2.resize(img, (256, 256))
    return img


class denoise_lterative:
    def __init__(self) -> None:
        self.t = 10

    def forward(self, image: np.ndarray) -> np.ndarray:
        delta = np.zeros(image.shape, dtype=np.float64)
        image_pad = np.pad(image, pad_width=1, mode="edge")
        operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        operator = operator * 1 / 10
        for i in range(0, self.height):
            for j in range(0, self.width):
                delta[i][j] = np.sum(image_pad[i : i + 3, j : j + 3] * operator)
        image = image + delta
        cv2.imshow("n_image", delta)
        cv2.waitKey(0)
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]
        for i in range(self.t):
            image = self.forward(image)
        return image


class denoise_analytic:
    def __init__(self) -> None:
        self.MAX = 10
        self.a = 1e-6
        self.t = 1e-6

    def cal_phi(self, image: np.ndarray) -> np.ndarray:
        def cal_phi_dot(nx, ny) -> float:
            result = 0.0
            for i in range(self.width):
                for j in range(self.height):
                    result += (
                        image[j][i]
                        * math.cos(math.pi * ny * i / self.width)
                        * math.cos(math.pi * nx * j / self.height)
                    )
            result = result * 4 / self.width / self.height
            return result

        phi = np.zeros((self.MAX, self.MAX), dtype=np.float32)
        for nx in range(1, self.MAX):
            for ny in range(1, self.MAX):
                phi[nx][ny] = cal_phi_dot(nx, ny)
        print(phi)
        return phi

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]
        phi = self.cal_phi(image)

        def cal_miu(nx, ny) -> float:
            return (math.pi * nx / self.height) ** 2 + (math.pi * ny / self.width) ** 2

        def cal_dot(x, y) -> int:
            result = 0.0
            for nx in range(1, self.MAX):
                for ny in range(1, self.MAX):
                    # print(cal_phi(nx, ny))
                    result += (
                        phi[nx][ny]
                        * math.exp(-self.a * cal_miu(nx, ny) * self.t)
                        * math.cos(math.pi * nx * x / self.height)
                        * math.cos(math.pi * ny * y / self.width)
                    )
            return max(0, int(result))

        denoised_image = np.zeros((self.width, self.height), dtype=np.uint8)
        for i in range(self.height):
            for j in range(self.width):
                dot = cal_dot(i, j)
                # assert dot < 0
                denoised_image[i][j] = dot
                # print(i, j, dot)
        return denoised_image


if __name__ == "__main__":
    images = get_images()
    # cv2.imshow("original_image", image2matrix(images[0]))
    # cv2.waitKey(0)
    # lter = denoise_analytic()
    # denoised_image = lter.denoise(image2matrix(images[0]))
    # cv2.imshow("denoised_image", denoised_image)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/chris/projects.localized/heat-denoise/data/de/denoised.jpeg", denoised_image)
    lter = denoise_lterative()
    denoised_image = lter.denoise(image2matrix(images[0]))
    # cv2.imshow("denoised_image", denoised_image)
    # cv2.waitKey(0)
