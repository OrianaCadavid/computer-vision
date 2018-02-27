import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import contrast_normalization

red_ch = cv2.imread('../images/red.jpg')
red_ch = red_ch[: , :, 0]
green_ch = cv2.imread('../images/green.jpg')
green_ch = green_ch[: , :, 0]
blue_ch = cv2.imread('../images/blue.jpg')
blue_ch= blue_ch[: , :, 0]

hist_red = cv2.calcHist([red_ch], [0], None, [256], [0, 256])
hist_green = cv2.calcHist([green_ch], [0], None, [256], [0, 256])
hist_blue = cv2.calcHist([blue_ch], [0], None, [256], [0, 256])

red_ch_norm = contrast_normalization(red_ch)
green_ch_norm = contrast_normalization(green_ch)
blue_ch_norm = contrast_normalization(blue_ch)

hist_red_norm = cv2.calcHist([red_ch_norm], [0], None, [256], [0, 256])
hist_green_norm = cv2.calcHist([green_ch_norm], [0], None, [256], [0, 256])
hist_blue_norm = cv2.calcHist([blue_ch_norm], [0], None, [256], [0, 256])

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.plot(hist_red, color='red')
ax1.plot(hist_green, color='green')
ax1.plot(hist_blue, color='blue')
ax2.plot(hist_red_norm, color='red')
ax2.plot(hist_green_norm, color='green')
ax2.plot(hist_blue_norm, color='blue')


img = cv2.merge([red_ch, green_ch, blue_ch])
img_norm = cv2.merge([red_ch_norm, green_ch_norm, blue_ch_norm])
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(img)
ax2.imshow(img_norm)
plt.show()
