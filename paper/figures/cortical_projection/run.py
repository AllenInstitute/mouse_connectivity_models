from shutil import copyfile
import os
import glob
import logging

import allensdk.core.json_utilities as ju

from helpers import get_projection_volume
from internal.scripts import create_top_views

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')
INPUT_JSON = os.path.join(TOP_DIR, 'input.json')


FILTERED = False
OUTPUT_DIR = 'output'
#OUTPUT_DIR = 'output' if FILTERED else 'unfiltered_output'
OUTPUT_DIR = os.path.join(FILE_DIR, OUTPUT_DIR)
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'post_processed_images')

INJECTION_REGIONS = ['MOp', 'VISp', 'SSp-m', 'SSp-ul', 'VISC', 'VISam', 'VISl', 'ACAd', 'ORBl', 'RSPv']
OUTPUT_IMAGE_FN = "blend_10.png"


def copy_image_to(region, projection_dir, image_dir, blend_fn):
    # NOTE : not the best

    s = lambda x: os.path.join(x, blend_fn)
    t = lambda x: os.path.join(image_dir, x.split('/')[-1] + ".png")

    # glob in case VISp or VISp_full
    dirs = glob.glob(os.path.join(projection_dir, region + "*"))

    map(lambda x: copyfile(s(x), t(x)), dirs)


def main():
    input_data = ju.read(INPUT_JSON)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    for region in INJECTION_REGIONS:

        # get projection volume
        logging.debug("="*40)
        logging.debug("-"*40)
        logging.debug("getting projection volume for %s" % region)
        logging.debug("-"*40)
        density_file = get_projection_volume.main(region, filtered=FILTERED)

        # create top view
        logging.debug("-"*40)
        logging.debug("creating top view for %s" % region)
        logging.debug("-"*40)
        output_directory = os.path.join(OUTPUT_DIR, 'top_views', region)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        create_top_views.main(density_file, output_directory)

        # copy to
        logging.debug("copying blend file")
        copy_image_to(region, output_directory, IMAGE_DIR, OUTPUT_IMAGE_FN)


if __name__ == "__main__":
    main()
