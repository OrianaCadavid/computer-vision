import cv2
import numpy as np


def contrast_normalization(channel, new_min_intensity=0, new_max_intensity=255, n=1):
    min_intesity = np.min(channel)
    max_intesity = np.max(channel)
    factor = (new_max_intensity - new_min_intensity) / (max_intesity - min_intesity)**n
    new_channel = factor * (channel - min_intesity)**n + new_min_intensity
    return new_channel.astype(np.uint8)


def stretchlim(channel, tol=0.05):
    height, width = channel.shape
    num_pixels = height * width
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cumsum = np.cumsum(hist / num_pixels, axis=0)
    index_min = np.argmax(cumsum > tol)
    index_max = np.argmax(cumsum > 1 - tol)
    return index_min, index_max
