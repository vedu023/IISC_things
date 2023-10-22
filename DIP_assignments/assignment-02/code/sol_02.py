from hashlib import new
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# nearest neighbor interpolation...
def nn(img, k):
    h,w = img.shape
    h1, w1 = int(h*k) ,int(w*k)

    d_img = np.zeros((h1,w1))
    for m in range(0, h1):
            for n in range(0, w1):
                i = int(m / k)
                j = int(n / k)
                d_img[m, n] = img[i, j]
    return d_img

# bilinear interpolation...
def bilinear(img, a1, b1):
    i1, j1 = a1, b1
    i2, j2 = a1+1, b1
    i3, j3 = a1, b1+1
    i4, j4 = a1+1, b1+1
    I = np.array([[img[i1, j1]], [img[i2, j2]], [img[i3, j3]], [img[i4, j4]]])
    ij = np.array([[1, i1, j1, i1*j1], [1, i2, j2, i2*j2], [1, i3, j3, i3*j3], [1, i4, j4, i4*j4]])

    A = np.matmul(np.linalg.pinv(ij), I)    # A0, A1, A2, A3 solution

    a = np.array([1, a1, b1, a1 * b1])
    value = np.matmul(a, A)                 # A0 + A1.i + A2.j + A3.ij -> here i = a1, j = a2

    return int(value[0])

# downsampling...
def downsample(img, k):
    h, w = img.shape
    h1, w1 = int(h/k), int(w/k)
    d_img = np.zeros([h1, w1])
    for m in range(0, h1):
        for n in range(0, w1):
            i = int(m*k)
            j = int(n*k)
            d_img[m, n] = img[i, j]
    return d_img

# upsampling....
def upsample(img, k, type = 0,):
    h, w = img.shape
    new_h = int(h * k)
    new_w = int(w * k)
    u_img = np.zeros((new_h, new_w))

    img = np.pad(img, ((1,1), (1,1)))

    if type == 0:    # for nearest neighbour interpolation
        u_img = nn(img, k)
        return u_img

    elif type == 1:   # for bilinear interpolation
        for m in range(0, new_h):
            for n in range(0, new_w):
                a1 = int(m / k) + 1   
                a2 = int(n / k) + 1  
                intensity = bilinear(img, a1, a2)
                u_img[m, n] = intensity
        return u_img
    

# error map...
def error_map(img, u_img):
    
    m,n  = u_img.shape
    img = np.pad(img, ((1,1), (1,1)))

    map = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            map[i,j] = (img[i,j] - u_img[i,j])**2
    return map


path = 'images/flowers.png'
img = io.imread(path)

d_img = downsample(img, 3)
u_img = upsample(d_img, 3, 1)

e_map = error_map(img, u_img)

fig, (ax1,ax2) = plt.subplots(2,2)

ax1[0].imshow(img, cmap='gray')
ax1[0].set_title("original image")

ax1[1].imshow(d_img, cmap='gray')
ax1[1].set_title("downsampled image")

ax2[0].imshow(u_img, cmap='gray')
ax2[0].set_title("upsampled image")

ax2[1].imshow(e_map, cmap='gray')
ax2[1].set_title("error map")

plt.show()

