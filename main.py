import os
import cv2
import numpy as np
from enum import Enum


def get_images() -> list:
    path = "data/original"
    images = [path + "/" + image for image in os.listdir(path)]
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


class DenoiseIterative:
    def __init__(self, a, t) -> None:
        self.t = t
        self.a = a

    def forward(self, image: np.ndarray) -> np.ndarray:
        delta = np.zeros(image.shape, dtype=np.float64)

        operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) * self.a
        delta = cv2.filter2D(image, -1, operator, borderType=cv2.BORDER_REPLICATE)
        img = image + delta
        img = np.clip(img, 0, 255)
        img = np.round(img).astype(np.uint8)
        return img

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]
        for i in range(self.t):
            image = self.forward(image)
        return image


class DenoiseAnalytic:
    def __init__(self, a, t, MAX) -> None:
        self.MAX = MAX
        self.a = a
        self.t = t
        self.nx = np.array(range(0, self.MAX))
        self.ny = np.array(range(0, self.MAX))

    def cal_phi(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]

        a = np.cos(np.pi * np.outer(self.nx, np.arange(0, self.height)) / self.height)
        b = np.cos(np.pi * np.outer(self.ny, np.arange(0, self.width)) / self.width)
        phi = np.einsum("ik,jl,kl->ij", a, b, image) * 4.0 / self.width / self.height
        phi[0, :] = phi[0, :] / 2
        phi[:, 0] = phi[:, 0] / 2

        return phi

    def cal_exp(self) -> np.ndarray:
        a = (np.pi * self.nx / self.height) ** 2
        b = (np.pi * self.ny / self.width) ** 2
        miu = np.add.outer(a, b)
        # print(miu.shape)
        exp = np.exp(-self.a * miu * self.t)
        # print("expsize", exp.shape)
        return exp

    def transform(self, original: np.ndarray, target: np.ndarray) -> np.ndarray:
        mean_original = np.mean(original)
        mean_target = np.mean(target)
        std_original = np.std(original)
        std_target = np.std(target)
        print(mean_original, mean_target, std_original, std_target)
        # transformed = (original - mean_original) * std_target / std_original + mean_target
        # transformed[transformed - mean_target > 3 * float(std_target)] = mean_target

        transformed = original

        # transformed = np.clip(transformed, 0, 255)
        transformed = np.round(transformed).astype(np.uint8)
        return transformed

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]

        exp = self.cal_exp()
        phi = self.cal_phi(image)
        # print(exp.shape, phi.shape)

        a = np.cos(np.pi * np.outer(self.nx, np.arange(0, self.height)) / self.height)
        b = np.cos(np.pi * np.outer(self.ny, np.arange(0, self.width)) / self.width)
        # print(a.shape, b.shape)
        u = np.einsum("ij,kl,ik,ik->jl", a, b, phi, exp)
        return self.transform(u, image)


class DenoiseType(Enum):
    ANALYTIC = 1
    ITERATIVE = 2


if __name__ == "__main__":
    # metadata
    d_type = DenoiseType.ANALYTIC
    target_size = 256
    a = 1
    t = 1e-1
    MAX = 1000
    lam = 10

    image_files = get_images()
    image = image2matrix(image_files[0], target_size)
    noisy_image = add_noise(image, lam)

    if d_type == DenoiseType.ANALYTIC:
        anal = DenoiseAnalytic(a, t, MAX)
        denoised_image = anal.denoise(noisy_image.copy())
    elif d_type == DenoiseType.ITERATIVE:
        iter_denoiser = DenoiseIterative(a=a, t=int(t))
        denoised_image = iter_denoiser.denoise(noisy_image)
    else:
        raise ValueError("DenoiseType is not defined")

    imgs = np.hstack((image, noisy_image, denoised_image))
    cv2.imwrite(f"data/results/{d_type.name}_a{a}_t{t}.png", imgs)
    cv2.imshow("imgs", imgs)
    cv2.waitKey(0)
