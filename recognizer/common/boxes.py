from dataclasses import dataclass, field

from typing import Tuple


@dataclass
class BoundaryBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = field(default=0.)
    bbox_id: int = field(init=False)

    def __post_init__(self):
        self.bbox_id = id(self)

    def __call__(self, img):
        """Crop image to box"""
        return img[self.y1: self.y2, self.x1: self.x2]

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def merge(self, bb: "BoundaryBox") -> "BoundaryBox":
        return BoundaryBox(
            top_left_x=min(self.x1, bb.x1),
            top_left_y=min(self.y1, bb.y1),
            bottom_right_x=max(self.x2, bb.x2),
            bottom_right_y=max(self.y2, bb.y2),
            confidence=max([self.confidence, bb.confidence]),
        )

    def box_is_inside_another(self, bb2, threshold=0.9) -> bool:
        intersection_area, bb1_area, bb2_area = self.get_boxes_intersection_area(
            other_box=bb2
        )
        if intersection_area == 0:
            return False
        return any((intersection_area / bb) > threshold for bb in (bb1_area, bb2_area))

    def get_boxes_intersection_area(self, other_box) -> Tuple:
        bb1 = self.box
        bb2 = other_box.box
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0.0
        else:
            intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
        bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
        return intersection_area, bb1_area, bb2_area

    def __getitem__(self, item):
        return self.box[item]
