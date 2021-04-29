import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import cv2 as cv

from recognizer.pipelines.augmentation import Distortion


@dataclass
class AtomVariantBase(ABC):
    atom: str

    @classmethod
    def get_image(cls, atom, text_renderer):
        return text_renderer.get_image(atom)

    @abstractmethod
    def draw(self):
        pass


class AtomNormal(AtomVariantBase):
    def draw(self, renderer):
        return self.get_image(self.atom, renderer)


class AtomVertical(AtomVariantBase):
    def draw(self, renderer):
        first_elem = self.get_image(self.atom[0], renderer)
        second_elem = self.get_image(self.atom[1], renderer)
        px = 2
        return cv.vconcat([first_elem[:-px], second_elem[px:]])


@dataclass
class AtomImageItem:
    classname: str
    variations: List

    def get_base_image(self, renderer):
        image = random.choice(self.variations)
        image = image.draw(renderer=renderer)
        params = [True, False]
        binary, dilated = random.choices(params, k=2)
        return Distortion.distort_image(image, binary, dilated)
