import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange
from matplotlib import pyplot as plt
import numba

class Seam:
    def __init__(self, filter_du, filter_dv, img):
        self.filter_du = filter_du
        self.filter_dv = filter_dv
        self.img = img


    def calc_energy(self):

        # print(self.img.shape)

        filter_du_3d = np.stack([self.filter_du] * 3, axis=2)
        filter_dv_3d = np.stack([self.filter_dv] * 3, axis=2)

        # print(filter_du_3d)
        # print(filter_dv_3d)

        self.img = self.img.astype('float32')
        convolved_x = np.absolute(convolve(self.img, filter_du_3d))
        convolved_y = np.absolute(convolve(self.img, filter_dv_3d))
        convolved = convolved_x + convolved_y

        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)
        # energy_map_x = convolved_x.sum(axis=2)
        # energy_map_y = convolved_y.sum(axis=2)

        return energy_map

    @numba.jit
    def minimum_seam(self):
        r, c, _ = self.img.shape
        energy_map = self.calc_energy()

        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        return M, backtrack


class Carve(Seam):
    def __init__(self, filter_du, filter_dv, img, scale):
        Seam.__init__(self, filter_du, filter_dv, img)
        self.scale = scale

    @numba.jit
    def carve_column(self):
        r, c, _ = self.img.shape

        M, backtrack = self.minimum_seam()
        mask = np.ones((r, c), dtype=np.bool)
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            # Mark the pixels for deletion
            mask[i, j] = False
            j = backtrack[i, j]

        mask = np.stack([mask] * 3, axis=2)
        self.img = self.img[mask].reshape((r, c - 1, 3))

        return self.img

    def reduceWidth(self):
        r, c, _ = self.img.shape
        new_c = c - self.scale

        for i in trange(c - new_c):  # use range if you don't want to use tqdm
            self.img = self.carve_column()

        return self.img

    def reduceHeight(self):
        self.img = np.rot90(self.img, 1, (0, 1))
        self.img = self.reduceWidth()
        self.img = np.rot90(self.img, 3, (0, 1))

        return self.img


class CarveFirstSeam(Seam):

    def __init__(self, filter_du, filter_dv, img, scale):
        Seam.__init__(self, filter_du, filter_dv, img)
        self.scale = scale

    @numba.jit        
    def show_seam_line(self):
       
        r, c, _ = self.img.shape
        M, backtrack = self.minimum_seam()
        mask = np.ones((r, c), dtype=np.bool)
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            mask[i, j] = False
            j = backtrack[i, j]

        mask = np.stack([mask] * 3, axis=2)

        for i in range(r):
            for j in range(c):
                # for k in range(3):
                # koitame mono to prwto stoixeio tis maskas mias kai einai idiow kai stis alles 2 diastaseis o pinakas
                if mask[i][j][0] == False:
                    self.img[i][j][0] = 255
                    self.img[i][j][1] = 0
                    self.img[i][j][2] = 0

        # imwrite(self.out_name, self.img)
        # plt.imshow(self.img)
        return self.img

    def showSeamWidth(self):
        r, c, _ = self.img.shape
        new_c = c - self.scale

        for i in trange(c - new_c):  # use range if you don't want to use tqdm
            self.img = self.show_seam_line()

        plt.imshow(self.img)
        return self.img

    def showSeamHeight(self):
        self.img = np.rot90(self.img, 1, (0, 1))
        self.img = self.showSeamWidth()
        self.img = np.rot90(self.img, 3, (0, 1))

        plt.imshow(self.img)
        return self.img