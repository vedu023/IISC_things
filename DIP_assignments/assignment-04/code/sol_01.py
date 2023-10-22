import skimage.io as io
import skimage.filters as flt
import skimage.transform as tr
import numpy as np
import matplotlib.pyplot as plt

path = 'Images/city.png'
img = io.imread(path)


def downsample(img, k):
    
    m,n = img.shape
    if not m/k==0 and n/k == 0:
        lp, rp = m%k, n%k

        img = np.pad(img, (lp,rp), constant_values=1)
        m,n = img.shape
    
    temp = np.zeros((m//k, n//k))
    for i in range(m//k):
        for j in range(n//k):
            temp[i,j] = img[i*k, j*k]

    return temp


def GLP(img, s, w, k):

    ks = w//2
    h = np.zeros((w,w))
    for i in range(-ks,ks):
        for j in range(-ks,ks):
            h[i,j] = np.exp(-((i)**2+(j)**2)/(2*s**2))
    sum_h = np.sum(h)
    h = h/sum_h

    img_gs = np.zeros(img.shape)
    img_p = np.pad(img, (ks,ks), constant_values=1)
    m,n = img_p.shape

    #apply filter...
    for i in range(w, m-w):
        for j in range(w, n-w):
            img_gs[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*h)
             
    #apply downsampling...
    img_n = downsample(img_gs,k)
    return img_n


def lib_fun(img,k):

    m,n = img.shape
    img = flt.gaussian(img)
    img = tr.resize(img, (m//k, n//k))
    return img


def mse(im1, im2):
    return np.sum((im1-im2)**2)


l = np.zeros(5)
img2 = lib_fun(img, 5)/255
for i in range(1,6):
    imgt = GLP(img, i, 5, 5)/255
    temp = mse(imgt,img2)
    l[i-1] = temp

m_mse, ms = np.min(l), np.argmin(l)
print(f'minimum mse : {m_mse} for sigma : {ms+1}')

img1 = GLP(img,ms,5,5)/255

fig1, (ax1,ax2, ax3) = plt.subplots(1,3)

ax1.imshow(downsample(img, 2), cmap='gray')
ax1.set_title("f-2")

ax2.imshow(downsample(img, 4), cmap='gray')
ax2.set_title("f-4")

ax3.imshow(downsample(img, 5), cmap='gray')
ax3.set_title("f-5")

plt.show()


fig2, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(img1, cmap='gray')
ax1.set_title("costum fun")
ax1.axis('off')

ax2.imshow(img2, cmap='gray')
ax2.set_title("lib fun")  
ax2.axis('off')

plt.show()

