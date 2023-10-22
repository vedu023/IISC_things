import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


path1 = 'images/lion.png'
path2 = 'images/StoneFace.png'
img1 = io.imread(path1)
img2 = io.imread(path2)

# Full Scale Contrast Stretching (FSCS)....
def fscs(img):
    img = np.array(img)
    k = 256

    A = np.min(img)
    B = np.max(img)
    N = B-A

    img_f =  ((k-1)/N) * (img - A)
    hist = np.histogram(img_f, bins=256)[0]

    return img_f, hist


# Histogram Equalization...
def hist_eq(img):

    m,n = img.shape
    eq_img = np.zeros((m,n))
    hist = np.histogram(img, bins=256)[0]

    cdf = np.cumsum(hist)
    for i in range(m):
        for j in range(n):
            eq_img[i,j] = cdf[img[i,j]]
    
    hist1 = np.histogram(eq_img, bins=256)[0]
    return eq_img, hist1


#Contrast Limited Adaptive Histogram Equalization (CLAHE):
def he_clip(img, clip):
    h, w = img.shape
    hist = np.histogram(img, bins = 256)[0]/(h*w)
    img_o = np.zeros([h, w])
    clip = np.max(hist)*clip
    extra = 0
    for i in range(256):
        if(hist[i] > clip):
            extra += hist[i] - clip
            hist[i] = clip
     
    e =  extra/256
    for i in range(256):
        hist[i] += e    

    new_cdf = np.cumsum(hist)

    for i in range(0, h):
        for j in range(0, w):
            img_o[i, j] = new_cdf[img[i, j]] * 255

    return img_o


# CLAHE without overlapping....
def clahe(img, clip):

    m, n = img.shape
    k, l = int(m/8), int(n/8)
    img_o = np.zeros(img.shape)

    l_cut = m % k  
    r_cut = n % l     
    h = m - l_cut   
    w = n - r_cut         
    img = img[:h, :w]

    for i in range(0, h, k):
        for j in range(0, w, l):
            img_o[i:i+k, j:j+l] = he_clip(img[i:i+k, j:j+l], clip)

    hist1 = np.histogram(img_o, bins=256)[0]
    return img_o, hist1

# CLAHE with overlap  
def clahe_overlap(img, clip):

    m, n = img.shape
    h = int(n/7.125)
    w = int(m/7.125)
    
    cs, rs = 0, 0
    ce, re = h, w

    img_0 = np.zeros((m, n))
    avg = np.zeros((m, n))     

    for _ in range(8):
        for _ in range(8):
            block = img[rs:re, cs:ce]
            block_he_output = he_clip(block, clip)
            img_0[rs:re, cs:ce] += block_he_output   
            avg[rs:re, cs:ce] += 1     
            cs = cs + int(0.875*h)            
            ce = cs + h
        cs = 0
        ce = cs + h                     
        rs = rs + int(0.875 * w)             
        re = rs + w

    temp = (avg == 0)    
    avg += temp         

    img_0 = img_0/avg                
    hist1 = np.histogram(img_0, bins=256)[0]
    return img_0, hist1


# Saturated contrast stretching...
def Scs(img):
    pass

img_01, hist1 = clahe_overlap(img1, 0.7)
img_02, hist2 = clahe_overlap(img2, 0.7)

fig1, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(img_01, cmap='gray')
ax2.imshow(img_02, cmap='gray')

fig2, (a1,a2) = plt.subplots(1,2)

a1.plot(hist1)
a1.set_title('histogram for CLAHE lion.png')

a2.plot(hist2)
a2.set_title('HE StonFace.png')

plt.show()
