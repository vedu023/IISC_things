import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

path1 = 'images/blur.png'
path2 = 'images/earth.png'
path3 = 'images/noisy.png'

b_img = io.imread(path1)
img = io.imread(path2)
n_img = io.imread(path3)

k1 = 1     # unsharp masking
k2 = 3     # high bost filtring...  k > 1

def hb_flt(img, b_img, k):
    img_g = np.zeros(img.shape)

    gmask = img - b_img
    img_g = b_img + (k1+1)*gmask

    return img_g


img_g = hb_flt(img, b_img, 1)
img_gn = hb_flt(img, n_img, 1)

fig, (ax1,ax2, ax3) = plt.subplots(1,3)

ax1.imshow(img, cmap='gray')
ax1.set_title("clean image")
ax1.axis('off')

ax2.imshow(img_g, cmap='gray')
ax2.set_title("blured image")
ax2.axis('off')

ax3.imshow(img_gn, cmap='gray')
ax3.set_title("noisy image")
ax3.axis('off')

plt.show()
 