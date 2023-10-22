import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

#input
img = io.imread('Images/circuitboard.tif')
m,n = img.shape
k_s = 5

img = np.pad(img, int(k_s/2), mode = 'constant')

kernel = np.ones((k_s,k_s))
win = np.zeros((k_s,k_s))

img_1 = np.zeros((m,n))
img_2 = np.zeros((m,n))
for i in range(m):
    for j in range(n):

        for k1 in range(k_s):
            for k2 in range(k_s):
                win[k1,k2] = img[k1+i,k2+j]*kernel[k1,k2]

        img_1[i,j] = np.sum(win)/(k_s**2) #mean fliter
        img_2[i,j] = np.median(win) #median filter


fig, (a1,a2) = plt.subplots(1,2)
a1.imshow(img_1, cmap = 'gray')
a1.set_title('Output of mean filter')
a1.axis('off')

a2.imshow(img_2, cmap = 'gray')
a2.set_title('Output of median filter')
a2.axis('off')
plt.show()