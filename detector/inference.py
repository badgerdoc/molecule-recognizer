import cv2
import logging
from pathlib import Path

from mmdet.apis import init_detector, inference_detector

from detector.utils import has_image_extension, extract_boxes_from_result

logger = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.7

CLASS_NAMES = ('atom', 'ring', 'SINGLE', 'DOUBLE', 'TRIPLE')


class CascadeRCNNInferenceService:
    def __init__(self, config: Path, model: Path, should_visualize: bool = False):
        self.model = init_detector(
            str(config.absolute()), str(model.absolute()), device='cpu'
        )
        self.should_visualize = should_visualize

    def inference_image(self, img_path: Path, out_path: Path, threshold: float = DEFAULT_THRESHOLD):
        if not has_image_extension(img_path):
            logger.warning(f"Not image {img_path}")
            return
        logger.info(f"Cascade inference image {img_path}")
        result = inference_detector(self.model, img_path)
        # if self.should_visualize:
        #     inference_image = self.model.show_result(img_path, result)
        #     cv2.imwrite(str(out_path.absolute()), inference_image)
        boxes = extract_boxes_from_result(result, CLASS_NAMES)
        img = cv2.imread(str(img_path))
        for box in boxes:
            if box['label'] == 'atom':
                x1, y1, x2, y2 = box['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.imwrite(str(out_path.absolute()), img)
