import numpy as np
import skimage.io as io
import skimage.color as clr

image1 = io.imread('img/IIScText.png')
image_bg = io.imread("img/IIScMainBuilding.png")

img_g = clr.rgb2gray(image1)

def threshold(img):
    
    hist, b = np.histogram(img, bins=256) # histogram
    
    b = b[:-1] 
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
 
    mu0 = np.cumsum(hist * b) / w0
    mu1 = (np.cumsum((hist * b)[::-1]) / w1[::-1])[::-1]

    s_b = w0 * w1 * (mu0 - mu1) ** 2  
    max_v = np.argmax(s_b)
    t = b[max_v]
    
    return t

t1 = threshold(img_g)

img_g[img_g >= t1] = 255
img_g[img_g < t1] = 0

image = clr.gray2rgb(img_g)

image_bg[np.where((image == [255,255,255]).all(axis=2))] = [0,125,125]

io.imshow(image_bg)
io.show()



