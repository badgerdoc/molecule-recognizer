import cv2
from typing import List

from recognizer.common.boxes import BoundaryBox


def draw_boxes(img, boxes: List[BoundaryBox], color, thk=1):
    img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thk)
    return img
