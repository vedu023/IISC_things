import numpy as np
import skimage.io as io
import skimage.color as clr
import skimage.filters as fls
import skimage.morphology as mrf
import skimage.transform as tr
import matplotlib.pyplot as plt


img = io.imread('Images/Checkerboard.png')
m,n = img.shape

ws = 3
ks = ws//2
h_response = np.zeros((m,n))


def modify(img):
    img = tr.rescale(img,2)
    img = tr.rotate(img,45)
    #g = np.random.normal(0, 1, (img.shape)) 
    return img 


def cs_img():
    s = 600
    img = np.zeros((s,s))
    img[300:,] = 1
    return img

# to get derivatives of img...
def sobel(img):

    #kernel
    Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])/8
    Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])/8

    img_s1 = np.zeros(img.shape)
    img_s2 = np.zeros(img.shape)
    img_p = np.pad(img, ks)
    m,n = img_p.shape

    for i in range(ws,m-ws):
        for j in range(ws,n-ws):

            img_s1[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gx)
            img_s2[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gy)

    return img_s1,img_s2


#stucture tensor setup
def harris(img):

    img_x, img_y = sobel(img)

    img_x = img_x**2
    img_y = img_y**2
    img_xy = img_x*img_y

    img_x = fls.gaussian(img_x)
    img_y = fls.gaussian(img_y)
    img_xy = fls.gaussian(img_xy)

    #Harris response calculation
    w = np.ones((3,3))/9
    for i in range(ws, m-ws):
        for j in range(ws, n-ws):

            img_x[i, j] = np.sum(img_x[i-ks:i+ks+1, j-ks:j+ks+1]*w)
            img_y[i, j] = np.sum(img_y[i-ks:i+ks+1, j-ks:j+ks+1]*w)
            img_xy[i, j] = np.sum(img_y[i-ks:i+ks+1, j-ks:j+ks+1]*w)

    k = 0.05

    detH = img_x*img_y - img_xy*img_xy
    traceH = img_x + img_y

    h_response = detH - k*traceH**2

    #for threshold ...
    h_response = mrf.dilation(h_response)
    max_h = np.max(h_response)
    tresh = 0.05*max_h

    #image with indicaed corners ...
    img_corners = clr.gray2rgb(np.copy(img))
    for i in range(m):
        for j in range(n):

            if h_response[i,j] > tresh:
                img_corners[i-2:i+2,j-2:j+2] = [255,0,0]
    
    return img_corners

plt.imshow(harris(img))
plt.show()

img = modify(img)
plt.imshow(harris(img))
plt.show()


 