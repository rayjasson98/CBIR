import sys

from PIL import Image

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


def concatenate_images(result_images, queried_image_name):
    images = [Image.open(x) for x in result_images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0

    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    new_image.save(f'result/retrieval_result/{queried_image_name}')


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
    result_images = []

    for match in result:
        print(f'{match["img"]}:\t{match["dis"]},\tClass {match["cls"]}')
        result_images.append(match["img"])

    split_img_path = img_path.split('/')
    queried_image_name = f'{mthd}/query_{split_img_path[1]}_{split_img_path[2]}'
    concatenate_images(result_images, queried_image_name)
