import cv2
import matplotlib.pyplot as plt
import numpy as np

red_ch = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 123, 0]
])
green_ch = np.array([
    [0, 0, 255],
    [0, 255, 255],
    [255, 123, 0]
])
blue_ch = np.array([
    [0, 255, 0],
    [0, 255, 255],
    [255, 123, 255]
])
img = cv2.merge([red_ch, green_ch, blue_ch])
img = img.astype(np.uint8)
plt.imshow(img)
plt.show()
