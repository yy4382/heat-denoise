import cv2
import numpy as np
from enum import Enum

__all__ = ["DenoiseType", "denoise_img", "DenoiseParams"]


class DenoiseType(Enum):
    ANALYTIC = 1
    ITERATIVE = 2


class DenoiseParams:
    def __init__(
        self, *, d_type: DenoiseType, a: float, t: int, MAX: int, lam: int, target_size: int = 256
    ) -> None:
        self.d_type = d_type
        self.a = a
        self.t = t
        self.MAX = MAX
        self.lam = lam
        self.target_size = target_size

    def __str__(self) -> str:
        basic = f"{self.d_type.name} a={self.a} t={self.t} lam={self.lam}"
        if self.d_type == DenoiseType.ANALYTIC:
            return f"{basic} infinity={self.MAX}"
        else:
            return basic


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
