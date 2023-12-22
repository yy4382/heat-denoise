import cv2
import numpy as np
import math


def Fourier_unfold(img,h,w,nxy=200):
    aux_x = np.arange(h)*np.pi/h
    cos_x = np.cos(aux_x.reshape(1,-1)*np.arange(0,nxy+1).reshape(-1,1))
    aux_y = np.arange(w)*np.pi/w
    cos_y = np.cos(aux_y.reshape(-1,1)*np.arange(0,nxy+1).reshape(1,-1))
    result = cos_x @ img @ cos_y
    result[0,:] /= 2
    result[:,0] /= 2
    return result/h/w*4


def denoise(img,phi,a=1,t=0,nxy=200):
    h,w = img.shape
    aux_x = (np.arange(0,nxy+1)*np.pi/h)**2
    aux_y = (np.arange(0,nxy+1)*np.pi/w)**2
    aux_ma = aux_x.reshape(-1,1) + aux_y.reshape(1,-1)
    exp_ma = np.exp(-a*t*aux_ma)
    coef_ma = phi * exp_ma

    aux_x2 = np.arange(nxy+1)*np.pi/h
    cos_x = np.cos(aux_x2.reshape(1, -1) * np.arange(0, h).reshape(-1, 1))
    aux_y2 = np.arange(nxy+1)*np.pi/w
    cos_y = np.cos(aux_y2.reshape(-1, 1) * np.arange(0, w).reshape(1, -1))
    new_img = cos_x @ coef_ma @ cos_y
    return new_img


class denoise_img:
    def __init__(self) -> None:
        self.MAX = 10
        self.a = 1e-6
        self.t = 1e-6

    def cal_phi(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[0]
        self.width = image.shape[1]
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
        return phi

    def denoise(self, image: np.ndarray) -> np.ndarray:
        self.height = image.shape[0]
        self.width = image.shape[1]
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

def image2matrix(image: str) -> np.ndarray:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    return img


def denoise2(img):
    std = float(np.std(img))
    mean = np.mean(img)
    img[np.abs(img-mean) > 3.0*std] = mean
    return img

def uniform(img,template=np.zeros(1)):
    img = denoise2(img)
    ma,mi = np.max(img),np.min(img)
    if template.any():
        mean,std = np.mean(template),np.std(template)
        img = (img-np.mean(img))/np.std(img)
        img = img*std+mean

    else:
        img = (img - mi) / (ma - mi) * 255
    return img.astype(np.uint8)


def add_noise(img,mean=0,std=5,p=0):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    mask = np.random.random(img.shape) < p
    noise *= mask
    noisy_img = cv2.add(img, noise)
    return noisy_img

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    img = image2matrix('data/original/Peppers.tif')
    noisy_img = add_noise(img)
    h,w = img.shape
    nxy = 10000
    phi= Fourier_unfold(noisy_img,h,w,nxy=nxy)
    # phi[np.abs(phi) < 1] = 0
    # phi[np.abs(phi) > 50] = 0
    new_img = denoise(noisy_img,phi,nxy=nxy)
    print(np.mean(new_img),np.mean(noisy_img),np.mean(img))
    cv2.imshow('new_img',cv2.resize(uniform(new_img),(512,512)))
    cv2.imshow('img',cv2.resize(noisy_img,(512,512)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()