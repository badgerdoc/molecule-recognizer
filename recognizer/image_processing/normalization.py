import cv2


def make_even_dimensions(img):
    img = img.copy()
    h, w = img.shape[:2]
    h += 1 if h % 2 == 1 else 0
    w += 1 if w % 2 == 1 else 0
    return cv2.resize(img, (w, h))
