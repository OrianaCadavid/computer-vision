import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../images/cartagena.jpg')
blue_ch, green_ch, red_ch = cv2.split(img)

hist_red = cv2.calcHist([red_ch], [0], None, [256], [0, 256])
hist_green = cv2.calcHist([green_ch], [0], None, [256], [0, 256])
hist_blue = cv2.calcHist([blue_ch], [0], None, [256], [0, 256])

pixels = np.arange(0, 256)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.bar(pixels, hist_red.ravel(), color='red')
ax2.bar(pixels, hist_green.ravel(), color='green')
ax3.bar(pixels, hist_blue.ravel(), color='blue')
plt.show()
