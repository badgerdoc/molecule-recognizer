import pytest
import cv2

from recognizer.image_processing.utils import find_content, norm_dims_base


@pytest.mark.parametrize('pad, shape', [
    (0, (102, 238, 3)),
    (3, (108, 244, 3)),
    (-3, (96, 232, 3)),
])
def test_find_content(pad, shape):
    img = cv2.imread('./resources/0b2a22e1090c.png')
    bbox = find_content(img, pad=pad)
    cropped_img_1 = bbox(img)
    assert cropped_img_1.shape == shape

    # FIXME: prevent cropped images from cropping any further, make 'find_content'
    #  idempotent.
    #  Test case below should pass too
    # bbox = find_content(img, pad=pad)
    # cropped_img_2 = bbox(cropped_img_1)
    # assert cropped_img_2.shape == shape


@pytest.mark.parametrize('base', [1, 2, 3, 8, 10])
def test_norm_dims_base(base):
    img = cv2.imread('./resources/0b2a22e1090c.png')
    norm_img = norm_dims_base(img, base)
    assert norm_img.shape[0] % base == 0
    assert norm_img.shape[1] % base == 0
