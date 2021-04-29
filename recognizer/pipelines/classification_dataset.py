import os
from multiprocessing import Process, Queue
from pathlib import Path

from recognizer.dataset_for_classifier.constants import ATOM_VARIATIONS
from recognizer.drawing.text import TextRenderer, TextRendererConfig
import cv2 as cv

conf = TextRendererConfig("fonts/Inconsolata-Regular.ttf")
text_renderer = TextRenderer(config=conf)


class DatasetForClassifierGeneratorPipeline:
    OUTPUT_PATH = Path(os.getcwd())

    def __init__(self,
                 train=1000,
                 test=100,
                 path=OUTPUT_PATH,
                 renderer=text_renderer,
                 processes=6):
        self.train = train
        self.test = test
        self.path = path
        self.renderer: TextRenderer = renderer
        self.processes = processes

    def create_class_sample(self, atom, sample_name, sample_size):
        path2class = f"{self.path}/images/{sample_name}/{atom.classname}"
        os.makedirs(path2class, exist_ok=True)

        for i in range(sample_size):
            img = atom.get_base_image(self.renderer)
            cv.imwrite(f"{path2class}/{i}.png", img)

    def create_set(self, set_name, set_size):
        queue = Queue()
        for atom in ATOM_VARIATIONS:
            queue.put((self.path, atom, set_name, set_size, self.renderer.config))
        proc = []
        while not queue.empty():
            if len(proc) < self.processes:
                p = Process(target=create_class_sample, args=(queue,))
                p.start()
                proc.append(p)
            proc = [p for p in proc if p.is_alive()]

    def create_dataset(self):
        self.create_set("train", self.train)
        self.create_set("test", self.test)


def create_class_sample(_input):
    path, atom, sample_name, sample_size, renderer_conf = _input.get()
    renderer = TextRenderer(renderer_conf)
    path2class = f"{path}/images/{sample_name}/{atom.classname}"
    os.makedirs(path2class, exist_ok=True)

    for i in range(sample_size):
        img = atom.get_base_image(renderer)
        cv.imwrite(f"{path2class}/{i}.png", img)


if __name__ == "__main__":
    d = DatasetForClassifierGeneratorPipeline()
    d.create_dataset()
