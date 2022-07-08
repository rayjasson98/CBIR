import itertools
import os

import numpy as np

from DB import Database
from evaluate import evaluate_class
from color import Color
from daisy import Daisy
from edge import Edge
from gabor import Gabor
from HOG import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
from glcm import GLCM


D_TYPE = 'd2'
DEPTH = 5
FEATURES = ['color', 'glcm']

RESULT_DIR = 'result'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class FeatureFusion():
    def __init__(self, features):
        assert len(features) > 1, "need to fuse more than one feature!"

        self.features = features
        self.samples = None

    def make_samples(self, db, verbose=False):
        if verbose:
            print(f'Use features {" & ".join(self.features)}')

        if self.samples == None:
            feats = []

            for f_class in self.features:
                feats.append(self._get_feat(db, f_class))

            samples = self._concat_feat(db, feats)
            self.samples = samples

        return self.samples

    def _get_feat(self, db, f_class):
        if f_class == 'color':
            f_c = Color()
        elif f_class == 'daisy':
            f_c = Daisy()
        elif f_class == 'edge':
            f_c = Edge()
        elif f_class == 'gabor':
            f_c = Gabor()
        elif f_class == 'hog':
            f_c = HOG()
        elif f_class == 'vgg':
            f_c = VGGNetFeat()
        elif f_class == 'res':
            f_c = ResNetFeat()
        elif f_class == 'glcm':
            f_c = GLCM()

        return f_c.make_samples(db, verbose=False)

    def _concat_feat(self, db, feats):
        samples = feats[0]
        delete_idx = []

        for idx in range(len(samples)):
            for feat in feats[1:]:
                feat = self._to_dict(feat)
                key = samples[idx]['img']

                if key not in feat:
                    delete_idx.append(idx)
                    continue

                assert feat[key]['cls'] == samples[idx]['cls']

                samples[idx]['hist'] = np.append(
                    samples[idx]['hist'], feat[key]['hist'])

        for d_idx in sorted(set(delete_idx), reverse=True):
            del samples[d_idx]

        if delete_idx != []:
            print(f'Ignore {len(set(delete_idx))} samples')

        return samples

    def _to_dict(self, feat):
        ret = {}

        for f in feat:
            ret[f['img']] = {
                'cls': f['cls'],
                'hist': f['hist']
            }

        return ret


def evaluate_feats(db, N, feat_pools=FEATURES, d_type='d1', depths=[None, 300, 200, 100, 50, 30, 10, 5, 3, 1]):
    result = open(os.path.join(
        RESULT_DIR, 'feature_fusion-{}-{}feats.csv'.format(d_type, N)), 'w')

    for i in range(N):
        result.write("feat{},".format(i))
    result.write("depth,distance,MMAP")
    combinations = itertools.combinations(feat_pools, N)

    for combination in combinations:
        fusion = FeatureFusion(features=list(combination))

        for d in depths:
            APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=d)
            cls_MAPs = []

            for cls, cls_APs in APs.items():
                MAP = np.mean(cls_APs)
                cls_MAPs.append(MAP)

            r = "{},{},{},{}".format(
                ",".join(combination), d, d_type, np.mean(cls_MAPs))
            print(r)

            result.write('\n'+r)

        print()

    result.close()


if __name__ == '__main__':
    db = Database()

    evaluate_feats(db, N=2, d_type='d1')

    fusion = FeatureFusion(features=['color', 'glcm'])
    class_APs = evaluate_class(
        db, f_instance=fusion, d_type=D_TYPE, depth=DEPTH)
    class_MAPs = []

    for klass, APs in class_APs.items():
        MAP = np.mean(APs)
        class_MAPs.append(MAP)

        print(f'Class: {klass}, MAP: {MAP}')

    print("MMAP", np.mean(class_MAPs))
