import cv2
import numpy as np
from enum import Enum

__all__ = ["DenoiseType", "denoise_img", "DenoiseParams", "NoiseType"]


class DenoiseType(Enum):
    ANALYTIC = 1
    ITERATIVE = 2


class NoiseType(Enum):
    GAUSS = 1
    POISSON = 2


class DenoiseParams:
    def __init__(
        self,
        *,
        d_type: DenoiseType,
        a: float,
        t: float,
        MAX: int,
        lam: int,
        target_size: int = 256,
        noise_type: NoiseType = NoiseType.GAUSS,
        gaussian_mean: float = 0,
        gaussian_sigma: float = 0.01,
        poisson_lam: int = 35,
    ) -> None:
        self.d_type = d_type
        self.a = a
        self.t = t
        self.MAX = MAX
        self.lam = lam
        self.target_size = target_size
        self.noise_type = noise_type
        self.gaussian_mean = gaussian_mean
        self.gaussian_sigma = gaussian_sigma
        self.poisson_lam = poisson_lam

    def __str__(self) -> str:
        basic = f"{self.d_type.name} a={self.a} t={self.t}"
        noise = f"noise_type={self.noise_type.name}"
        if self.noise_type == NoiseType.GAUSS:
            noise += f" mean={self.gaussian_mean} sigma={self.gaussian_sigma}"
        elif self.noise_type == NoiseType.POISSON:
            noise += f" lam={self.poisson_lam}"
        if self.d_type == DenoiseType.ANALYTIC:
            return f"{basic} {noise} infinity={self.MAX}"
        else:
            return basic + " " + noise


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
        # phi = np.einsum("ik,jl,kl->ij", a, b, image) * 4.0 / self.width / self.height
        phi = a @ image @ b.T * 4.0 / self.width / self.height
        phi[0, :] = phi[0, :] / 2
        phi[:, 0] = phi[:, 0] / 2

        return phi

    def cal_exp(self) -> np.ndarray:
        a = (np.pi * self.nx / self.height) ** 2
        b = (np.pi * self.ny / self.width) ** 2
        miu = np.add.outer(a, b)
        exp = np.exp(-self.a * miu * self.t)
        return exp

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[1]
        self.width = image.shape[0]

        exp = self.cal_exp()
        phi = self.cal_phi(image)

        a = np.cos(np.pi * np.outer(self.nx, np.arange(0, self.height)) / self.height)
        b = np.cos(np.pi * np.outer(self.ny, np.arange(0, self.width)) / self.width)
        # u = np.einsum("ij,kl,ik,ik->jl", a, b, phi, exp)
        u = a.T @ (phi * exp) @ b
        return np.round(u).astype(np.uint8)


def denoise_img(noisy_image: np.ndarray, params: DenoiseParams) -> np.ndarray:
    d_type = params.d_type
    a = params.a
    t = params.t
    MAX = params.MAX
    if d_type == DenoiseType.ANALYTIC:
        anal = DenoiseAnalytic(a, t, MAX)
        denoised_image = anal.denoise(noisy_image.copy())
    elif d_type == DenoiseType.ITERATIVE:
        iter_denoiser = DenoiseIterative(a=a, t=int(t))
        denoised_image = iter_denoiser.denoise(noisy_image)
    else:
        raise ValueError("DenoiseType is not defined")
    return denoised_image
