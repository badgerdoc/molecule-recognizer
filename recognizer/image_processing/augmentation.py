import cv2
import numpy as np


def apply_binary_threshold(img, threshold):
    img = img.copy()
    ret, thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh1


def preprocess_original(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return apply_binary_threshold(gray, 190)


def add_noise(img, color, percent):
    out_img = img.copy()
    noise = np.random.rand(*out_img.shape[:2])
    noise_mask = noise > percent
    out_img[noise_mask] = color
    return out_img