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
QUERY_IDX = 0

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
    except IndexError:
        print(f'usage: {sys.argv[0]} <method>')
        print("supported methods:\ncolor, daisy, edge, gabor, hog, vgg, resnet, glcm")

        sys.exit(1)

    samples = getattr(methods[mthd](), "make_samples")(db)

    query = samples[QUERY_IDX]
    print(f'\n[+] query: {query["img"]}\n')

    _, result = infer(query, samples=samples, depth=DEPTH, d_type=D_TYPE)

    for match in result:
        print(f'{match["dis"]},\tClass {match["cls"]}')
