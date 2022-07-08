import os
import pickle

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import color

from DB import Database
from evaluate import evaluate_class

N_SLICE = 3
H_TYPE = 'region'
D_TYPE = 'd2'
DEPTH = 5

CACHE_DIR = 'cache'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class GLCM():
    def histogram(self, input, h_type=H_TYPE, n_slice=N_SLICE):
        if isinstance(input, np.ndarray):
            img = input.copy()
        else:
            img = cv2.imread(input, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape

        if h_type == 'global':
            glcm_props = self.glcm(img)
        elif h_type == 'region':
            glcm_props = np.zeros((n_slice, n_slice, 12))
            h_slice = np.around(np.linspace(
                0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(
                0, width, n_slice+1, endpoint=True)).astype(int)

            for hs in range(len(h_slice)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_slice[hs]:h_slice[hs+1],
                                w_slice[ws]:w_slice[ws+1]]
                    glcm_props[hs][ws] = self.glcm(img_r)

        return glcm_props.flatten()

    def glcm(self, img):
        image = np.uint8(color.rgb2gray(img) * 255)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2]
        glcm = graycomatrix(image, distances=distances, angles=angles,
                            levels=256, symmetric=True, normed=True)
        glcm_props = np.zeros((len(distances), len(angles), 4))

        for i in range(len(distances)):
            for j in range(len(angles)):
                contrast = graycoprops(glcm, 'contrast')[i, j]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[i, j]
                homogeneity = graycoprops(glcm, 'homogeneity')[i, j]
                correlation = graycoprops(glcm, 'correlation')[i, j]

                glcm_props[i, j] = [
                    contrast, dissimilarity, homogeneity, correlation]

        return glcm_props.flatten()

    def make_samples(self, db, verbose=True):
        if H_TYPE == 'global':
            sample_cache = f'glcm-{H_TYPE}-{D_TYPE}-{DEPTH}'
        elif H_TYPE == 'region':
            sample_cache = f'glcm-nslice{N_SLICE}-{H_TYPE}-{D_TYPE}-{DEPTH}'

        try:
            samples = pickle.load(
                open(os.path.join(CACHE_DIR, sample_cache), "rb", True))

            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])

            if verbose:
                print('Using cache..., ',
                      f'config={sample_cache}, ',
                      f' distance={D_TYPE}, ',
                      f'depth={DEPTH}')
        except FileNotFoundError:
            if verbose:
                print('Counting histograms..., ',
                      f'config={sample_cache}, ',
                      f' distance={D_TYPE}, ',
                      f'depth={DEPTH}')

            samples = []
            data = db.get_data()

            for d in data.itertuples():
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_hist = self.histogram(d_img, h_type=H_TYPE, n_slice=N_SLICE)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_hist
                })

            pickle.dump(samples,
                        open(os.path.join(CACHE_DIR, sample_cache), "wb", True))

        return samples


if __name__ == '__main__':
    db = Database()

    class_APs = evaluate_class(db, f_class=GLCM, d_type=D_TYPE, depth=DEPTH)
    class_MAPs = []

    for klass, APs in class_APs.items():
        MAP = np.mean(APs)
        class_MAPs.append(MAP)

        print(f'Class: {klass}, MAP: {MAP}')

    print("MMAP", np.mean(class_MAPs))
