import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import imutils
from scipy import signal


img = cv2.imread('test2.png')



# First a 1-D  Gaussian
t = np.linspace(-10, 10, 30)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]


# # mode='same' is there to enforce the same output shape as input arrays
# # (ie avoid border effects)
img_cv = signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')

cv2.imwrite("cv2.png",img_cv)


cv2.imshow("img_cv",img_cv.clip(0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()