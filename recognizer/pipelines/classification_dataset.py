import os
from pathlib import Path

import cv2 as cv

from recognizer.dataset_for_classifier.dataset_creation import (
    AtomImageItem,
    AtomNormal,
    AtomVertical,
)
from recognizer.drawing.text import TextRenderer, TextRendererConfig

ATOMS = (
    AtomImageItem(
        "NH",
        [AtomNormal("NH"),
         AtomNormal("HN"),
         AtomVertical("NH"),
         AtomVertical("HN")],
    ),
    AtomImageItem("OH", [AtomNormal("OH"), AtomNormal("HO")]),
    AtomImageItem("Cl_[3]C", [AtomNormal("CCl_[3]"), AtomNormal("Cl_[3]C")]),
    AtomImageItem("F_[3]C", [AtomNormal("F_[3]C"), AtomNormal("CF_[3]")]),
    AtomImageItem("H_[3]Si", [AtomNormal("H_[3]Si"), AtomNormal("SiH_[3]")]),
    AtomImageItem("HS", [AtomNormal("HS"), AtomNormal("SH")]),
    AtomImageItem("NC", [AtomNormal("NC"), AtomNormal("CN")]),
    AtomImageItem("BH_[2]", [AtomNormal("BH_[2]")]),
    AtomImageItem("Br", [AtomNormal("Br")]),
    AtomImageItem("C_[2]H_[5]", [AtomNormal("C_[2]H_[5]")]),
    AtomImageItem("CH", [AtomNormal("CH")]),
    AtomImageItem("CH_[2]", [AtomNormal("CH_[2]")]),
    AtomImageItem("CH_[3]", [AtomNormal("CH_[3]")]),
    AtomImageItem("Cl", [AtomNormal("Cl")]),
    AtomImageItem("F", [AtomNormal("F")]),
    AtomImageItem("H", [AtomNormal("H")]),
    AtomImageItem("I", [AtomNormal("I")]),
    AtomImageItem("N_[3]", [AtomNormal("N_[3]")]),
    AtomImageItem("NH_[2]", [AtomNormal("NH_[2]")]),
    AtomImageItem("NO", [AtomNormal("NO")]),
    AtomImageItem("NO_[2]", [AtomNormal("NO_[2]")]),
    AtomImageItem("O", [AtomNormal("O")]),
    AtomImageItem("S", [AtomNormal("S")]),
    AtomImageItem("Si", [AtomNormal("Si")]),
)

conf = TextRendererConfig("../../fonts/Inconsolata-Regular.ttf")
text_renderer = TextRenderer(config=conf)


class DatasetForClassifierGeneratorPipeline:
    OUTPUT_PATH = Path(os.getcwd())

    def __init__(self,
                 train=200,
                 test=50,
                 path=OUTPUT_PATH,
                 renderer=text_renderer):
        self.train = train
        self.test = test
        self.path = path
        self.renderer = renderer

    def create_class_sample(self, atom, sample_name, sample_size):
        path2class = f"{self.path}/images/{sample_name}/{atom.classname}"
        os.makedirs(path2class, exist_ok=True)

        for i in range(sample_size):
            img = atom.get_base_image(self.renderer)
            cv.imwrite(f"{path2class}/{i}.png", img)

    def create_set(self, set_name, set_size):
        for atom in ATOMS:
            self.create_class_sample(atom, set_name, set_size)

    def create_dataset(self):
        self.create_set("train", self.train)
        self.create_set("test", self.test)


if __name__ == "__main__":
    d = DatasetForClassifierGeneratorPipeline()
    d.create_dataset()
