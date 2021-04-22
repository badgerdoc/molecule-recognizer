import os
from pathlib import Path
from random import uniform, randint
import logging

from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import pandas as pd

from recognizer.pipelines.augmentation import Distortion

log = logging.getLogger(__name__)

ATOMS_CLASSES = (
    "Cl",
    "H",
    "C",
    "O",
    "N",
    "F",
    "Br",
    "S",
    "Si"
    )


class NoFontException(Exception):
    def __str__(self):
        return self.__doc__


class AtomImageGeneratorPipeline:
    FONTS_PATH = Path(os.getcwd()) / "fonts"
    OUTPUT_PATH = Path(os.getcwd()) / "atoms_images"

    def __init__(
        self,
        image_height_width=(17, 17),
        range_text_size=(13, 15),
        output_names=("train.csv", "test.csv", "val.csv")
    ):

        self.image_height_width = image_height_width
        self.range_text_size = range_text_size
        self.output_names = output_names
        self._make_output_dir()

    def process_batch(self, fonts_amount=None, atoms_list=ATOMS_CLASSES):
        fonts = [font for font in self.FONTS_PATH.iterdir()]
        if not len(fonts):
            raise NoFontException()

        all_pictures_data = {}

        fonts_amount = fonts_amount or len(fonts)

        fonts = np.random.permutation(fonts)

        for font in fonts[:fonts_amount]:
            data = self.process_item(font, atoms_list)
            all_pictures_data.update(data)

        self._save_data_to_csv(all_pictures_data)

    def process_item(self, path_to_font, atoms_list):
        batch_pictures_data = {}

        for atom in atoms_list:
            image = Image.new("L", self.image_height_width, color=255)
            draw = ImageDraw.Draw(image)

            text_size = randint(*self.range_text_size)
            picture_font = ImageFont.truetype(str(path_to_font), text_size)
            rand_offset = round(uniform(0, 0.5), 1)

            draw.text((rand_offset, 0), atom, font=picture_font, fill=0)

            picture_hash = hash(
                f"{path_to_font.stem}_{atom}_{text_size}_{rand_offset}"
                )

            batch_pictures_data[picture_hash] = atom
            self._save_image(picture_hash, image)

        return batch_pictures_data

    def _save_image(self, picture_name, image):
        image = np.array(image)
        image = Distortion().distort_image(image, binary=False, dilated=True)
        cv2.imwrite(
            f"{str(self.OUTPUT_PATH)}/{picture_name}.png",
            image
        )

    def _save_data_to_csv(self, pic_data):
        train_test_val = self.split_train_test_val(pic_data)

        for indx, images_list in enumerate(train_test_val):
            data = {"image": [], "label": []}
            for img in images_list:
                data["image"].append(img)
                data["label"].append(pic_data[img])

            pd.DataFrame(data).to_csv(
                f"{str(self.OUTPUT_PATH)}/{self.output_names[indx]}"
                )

    @staticmethod
    def split_train_test_val(pictures):
        rand_list_of_pictures = np.random.permutation(list(pictures.keys()))
        train = round(.7 * len(rand_list_of_pictures))
        test = round(.2 * len(rand_list_of_pictures))

        return (
            rand_list_of_pictures[:train],
            rand_list_of_pictures[train:train+test],
            rand_list_of_pictures[train+test:]
            )

    def _make_output_dir(self):
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)


if __name__ == '__main__':
    p = AtomImageGeneratorPipeline()
    p.process_batch()
