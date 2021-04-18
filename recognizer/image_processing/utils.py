import cv2
from typing import Tuple, Union

from recognizer.common.boxes import BoundaryBox

DEFAULT_BG_VALUE = 255
DEFAULT_PADDING = 3


def bgr2grayscale(img):
    img = img.copy()
    if len(img.shape) == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_bgr(img):
    img = img.copy()
    if len(img.shape) == 3 and img.shape[2] > 1:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _adjust_bg_value(img, bg) -> Union[Tuple[int, int, int], int]:
    if len(img.shape) == 3 and img.shape[2] > 1:
        return bg, bg, bg
    return bg


def to_binary_img_naive(img, bg=DEFAULT_BG_VALUE):
    img = img.copy()
    img = bgr2grayscale(img)
    mask = img == bg
    img[mask] = bg
    mask = img != bg
    img[mask] = 0
    return img


def find_content(img, bg=DEFAULT_BG_VALUE, pad=DEFAULT_PADDING) -> BoundaryBox:
    """Get boundary box of non-background pixels"""
    binary_img = to_binary_img_naive(img=img, bg=bg)
    h, w = img.shape[:2]
    y1, y2 = h + 1, -1
    x1, x2 = w + 1, -1
    for y in range(h):
        for x in range(w):
            if binary_img[y][x] != bg:
                y1 = y if y < y1 else y1
                y2 = y if y > y2 else y2
                x1 = x if x < x1 else x1
                x2 = x if x > x2 else x2
    y1p = y1 - pad if y1 - pad >= 0 else 0
    x1p = x1 - pad if x1 - pad >= 0 else 0
    y2p = y2 + pad if y2 + pad <= h else h
    x2p = x2 + pad if x2 + pad <= w else w
    return BoundaryBox(x1p, y1p, x2p, y2p)


def extract_content(img, bg=DEFAULT_BG_VALUE, pad=DEFAULT_PADDING):
    return find_content(img, bg, pad)(img)


def resize_with_padding(img, size: Tuple[int, int], bg=DEFAULT_BG_VALUE):
    """
    Use padding to get image of desired size.
    :param img: Source image
    :param size: Height and width
    :param bg: Fill color
    :return:
    """
    img = img.copy()
    bg = _adjust_bg_value(img, bg)
    h, w = img.shape[:2]
    _h, _w = size
    dh, dw = _h - h, _w - w
    if dh < 0 or dw < 0:
        raise ValueError(f'Can not resize image with shape {(h, w)} to {size}.')
    vpad, vmod = divmod(dh, 2)
    hpad, hmod = divmod(dw, 2)
    return cv2.copyMakeBorder(
        img,
        vpad,
        vpad + vmod,
        hpad,
        hpad + hmod,
        borderType=cv2.BORDER_CONSTANT,
        value=bg
    )


def norm_dims_base(img, base, bg=DEFAULT_BG_VALUE):
    """
    Resize image to achieve specific dimensions (e.g. width and height have to be
    product of 8)
    """
    if base <= 0:
        raise ValueError(f'Base value {base} should be greater than zero.')
    img = img.copy()
    h, w = img.shape[:2]
    size = ((h // base + 1) * base, (w // base + 1) * base)
    return resize_with_padding(img, size, bg)
