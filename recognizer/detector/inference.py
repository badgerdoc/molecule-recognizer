import cv2
import logging
from pathlib import Path

from mmdet.apis import init_detector, inference_detector

from recognizer.common.constants import ATOM_CLS, DOUBLE_CLS, RING_CLS, \
    SINGLE_CLS, TRIPLE_CLS
from recognizer.detector.structure import DetectedStructure
from recognizer.detector.utils import validate_image_extension, \
    extract_boxes_from_result
from recognizer.drawing.boxes import draw_boxes

logger = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.7

CLASS_NAMES = (ATOM_CLS, RING_CLS, SINGLE_CLS, DOUBLE_CLS, TRIPLE_CLS)


class CascadeRCNNInferenceService:
    def __init__(
        self, config: Path, model: Path, should_visualize: bool = False
    ):
        self.model = init_detector(
            str(config.absolute()), str(model.absolute()), device='cuda'
        )
        self.should_visualize = should_visualize

    def inference_image(
        self,
        img,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> DetectedStructure:
        result = inference_detector(self.model, img)
        boxes = extract_boxes_from_result(result, CLASS_NAMES, threshold)
        structure = DetectedStructure.from_bboxes_list(boxes)
        return structure

    @staticmethod
    def visualize_boxes(
        structure: DetectedStructure, img, out_path: Path
    ):
        img = img.copy()
        img = draw_boxes(img, [a.bbox for a in structure.atoms], (255, 0, 0))
        img = draw_boxes(img, [b.bbox for b in structure.bonds], (0, 255, 0))
        cv2.imwrite(str(out_path.absolute()), img)
