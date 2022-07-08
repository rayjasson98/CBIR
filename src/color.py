import itertools
import os
import pickle

import cv2
import numpy as np

from DB import Database
from evaluate import evaluate_class

N_BIN = 12
N_SLICE = 3
H_TYPE = 'region'
D_TYPE = 'd2'
DEPTH = 5

CACHE_DIR = 'cache'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class Color():
    def histogram(self, input, n_bin=N_BIN, h_type=H_TYPE, n_slice=N_SLICE, normalize=True):
        ''' count img color histogram

          arguments
            input    : a path to a image or a numpy.ndarray
            n_bin    : number of bins for each channel
            type     : 'global' means count the histogram for whole image
                       'region' means count the histogram for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output histogram

          return
            type == 'global'
              a numpy array with size n_bin ** channel
            type == 'region'
              a numpy array with size n_slice * n_slice * (n_bin ** channel)
        '''
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = cv2.imread(input, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channel = img.shape
        # slice bins equally for each channel
        bins = np.linspace(0, 256, n_bin+1, endpoint=True)

        if h_type == 'global':
            hist = self._count_hist(img, n_bin, bins, channel)
        elif h_type == 'region':
            hist = np.zeros((n_slice, n_slice, n_bin ** channel))
            h_slice = np.around(np.linspace(
                0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(
                0, width, n_slice+1, endpoint=True)).astype(int)

            for hs in range(len(h_slice)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_slice[hs]:h_slice[hs+1],
                                w_slice[ws]:w_slice[ws+1]]
                    hist[hs][ws] = self._count_hist(img_r, n_bin,
                                                    bins, channel)

        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _count_hist(self, input, n_bin, bins, channel):
        img = input.copy()

        # Permutation of bins
        bins_idx = {key: idx for idx, key in
                    enumerate(itertools.product(np.arange(n_bin), repeat=channel))}
        hist = np.zeros(n_bin ** channel)

        # Cluster every pixels
        for idx in range(len(bins)-1):
            img[(input >= bins[idx]) & (input < bins[idx+1])] = idx

        # Add pixels into bins
        height, width, _ = img.shape

        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h, w])]
                hist[b_idx] += 1

        return hist

    def make_samples(self, db, verbose=True):
        if H_TYPE == 'global':
            sample_cache = "histogram_cache-{}-n_bin{}".format(H_TYPE, N_BIN)
        elif H_TYPE == 'region':
            sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(
                H_TYPE, N_BIN, N_SLICE)

        try:
            samples = pickle.load(
                open(os.path.join(CACHE_DIR, sample_cache), "rb", True))
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
                d_hist = self.histogram(
                    d_img, h_type=H_TYPE, n_bin=N_BIN, n_slice=N_SLICE)
                samples.append({
                    'img':  d_img,
                    'cls':  d_cls,
                    'hist': d_hist
                })

            pickle.dump(samples, open(os.path.join(
                CACHE_DIR, sample_cache), "wb", True))

        return samples


if __name__ == '__main__':
    db = Database()

    class_APs = evaluate_class(db, f_class=Color, d_type=D_TYPE, depth=DEPTH)
    class_MAPs = []

    for klass, APs in class_APs.items():
        MAP = np.mean(APs)
        class_MAPs.append(MAP)

        print(f'Class: {klass}, MAP: {MAP}')

    print("MMAP", np.mean(class_MAPs))
