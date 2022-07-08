import sys

from color import Color
from daisy import Daisy
from DB import Database
from edge import Edge
from evaluate import infer
from gabor import Gabor
from HOG import HOG
from resnet import ResNetFeat
from vggnet import VGGNetFeat
from glcm import GLCM

DEPTH = 5
D_TYPE = 'd1'

if __name__ == '__main__':
    db = Database()

    methods = {
        "color": Color,
        "daisy": Daisy,
        "edge": Edge,
        "hog": HOG,
        "gabor": Gabor,
        "vgg": VGGNetFeat,
        "resnet": ResNetFeat,
        "glcm": GLCM
    }

    try:
        mthd = sys.argv[1].lower()
        img_path = sys.argv[2].lower()
    except IndexError:
        print(f'usage: {sys.argv[0]} <method> <img_path>')
        print("supported methods: color, daisy, edge, gabor, hog, vgg, resnet, glcm")

        sys.exit(1)

    samples = getattr(methods[mthd](), "make_samples")(db)
    query = next((sample for sample in samples if sample['img'] == img_path),
                 None)

    if query is None:
        print(f'{img_path} not found in database')
        sys.exit(1)

    print(f'\n[+] query: {query["img"]}\n')

    _, result = infer(query, samples=samples, depth=DEPTH, d_type=D_TYPE)

    for match in result:
        print(f'{match["img"]}:\t{match["dis"]},\tClass {match["cls"]}')
