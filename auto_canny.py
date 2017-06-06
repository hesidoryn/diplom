import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('test.png',0)
edges = cv2.Laplacian(img,cv2.CV_64F)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# from PIL import Image
# image = Image.open('test.png')
# mask=image.convert("L")
# mask.save('test12312.png')
# th=190 # the value has to be adjusted for an image of interest 
# mask = mask.point(lambda i: i < th and 255)
# mask.save('test123123.png')

# import numpy
# import scipy
# from scipy import ndimage
# 
# im = scipy.misc.imread('test12312.png')
# im = im.astype('int32')
# dx = ndimage.sobel(im, 0)  # horizontal derivative
# dy = ndimage.sobel(im, 1)  # vertical derivative
# mag = numpy.hypot(dx, dy)  # magnitude
# mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
# scipy.misc.imsave('sobel.png', mag)

# from PIL import ImageFilter
# 
# im = image.filter(ImageFilter.CONTOUR)
# im.save('test_ne1x3.png')
# 
# im2 = im.filter(ImageFilter.MinFilter(3))
# im2.save('test_ne1x4.png')
# im3 = im.filter(ImageFilter.MinFilter)  # same as MinFilter(3)
# im3.save('test_ne1x5.png')