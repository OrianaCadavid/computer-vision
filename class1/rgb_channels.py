import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../images/cartagena.jpg')
blue_ch, green_ch, red_ch = cv2.split(img)

img_rgb = cv2.merge([red_ch, green_ch, blue_ch])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Separate channels')
ax1.imshow(img_rgb)
ax1.set_title('Original')
ax2.imshow(red_ch, cmap='gray')
ax2.set_title('Red channel')
ax3.imshow(blue_ch, cmap='gray')
ax3.set_title('Blue channel')
ax4.imshow(green_ch, cmap='gray')
ax4.set_title('Green channel')

img_red = np.copy(img_rgb)
img_red[:, :, 1] = 0
img_red[:, :, 2] = 0

img_green = np.copy(img_rgb)
img_green[:, :, 0] = 0
img_green[:, :, 2] = 0

img_blue = np.copy(img_rgb)
img_blue[:, :, 0] = 0
img_blue[:, :, 1] = 0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Separate channels as images')
ax1.imshow(img_rgb)
ax1.set_title('Original')
ax2.imshow(img_red)
ax2.set_title('Only red channel')
ax3.imshow(img_green)
ax3.set_title('Only blue channel')
ax4.imshow(img_blue)
ax4.set_title('Only green channel')
plt.show()
