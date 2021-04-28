from recognizer.classifier.fastai_classifier import FastaiAtomClassifier
from recognizer.common.boxes import BoundaryBox
from recognizer.detector.inference import CascadeRCNNInferenceService
from recognizer.drawing.text import TextRenderer
from recognizer.image_processing.utils import bgr2grayscale


BG_COLOR = 255


class TextRestorationService:
    def __init__(
        self,
        classifier: FastaiAtomClassifier,
        detector: CascadeRCNNInferenceService,
        renderer: TextRenderer,
    ):
        self.classifier = classifier
        self.detector = detector
        self.renderer = renderer

    @staticmethod
    def clear_area(img, bbox: BoundaryBox):
        x1, y1, x2, y2 = bbox.box
        img[y1: y2, x1: x2] = BG_COLOR
        return img

    def restore(self, ref_img, img_to_restore):
        ref_img = ref_img.copy()
        img_to_restore = bgr2grayscale(img_to_restore)
        img_to_restore = img_to_restore.copy()
        atoms = self.detector.inference_image(ref_img).atoms
        for atom in atoms:
            roi = atom.bbox(ref_img)
            atom_cls = self.classifier.predict(roi)
            img_to_restore = self.draw_text(img_to_restore, atom.bbox, atom_cls)
        return img_to_restore

    def draw_text(self, img, bbox: BoundaryBox, text: str):
        img = img.copy()
        img = self.clear_area(img, bbox)
        text_img = self.renderer.get_image(text)
        x0, y0 = bbox.box[:2]
        th, tw = text_img.shape[:2]
        img[y0: y0+th, x0: x0+tw] = text_img
        return img
