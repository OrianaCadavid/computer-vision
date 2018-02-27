import cv2
import matplotlib.pyplot as plt
import numpy as np


def bgr2rgb(img_bgr):
    '''
    Converts an image from BGR to RGB
    '''
    blue_ch, green_ch, red_ch = cv2.split(img_bgr)
    img_rgb = cv2.merge([red_ch, green_ch, blue_ch])
    return img_rgb


boat = cv2.imread('../images/boat.jpg')
beach = cv2.imread('../images/beach.jpg')
boat = bgr2rgb(boat)
beach = bgr2rgb(beach)

factor = 0.5
added_images = boat * factor + beach * (1 - factor)
added_images = added_images.astype(np.uint8)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.imshow(boat)
ax2.imshow(beach)
ax3.imshow(added_images)
plt.show()
