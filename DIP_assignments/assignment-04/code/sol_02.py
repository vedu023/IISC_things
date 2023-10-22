import numpy as np
import skimage.io as io
import skimage.transform as tr
import  matplotlib.pyplot as plt



img1 = io.imread('Images/Checkerboard.png')
img2 = io.imread('Images/NoisyCheckerboard.png')
img3 = io.imread('Images/Coins.png')
img4 = io.imread('Images/NoisyCoins.png')

sigma = 5
ws = 5

def g_smoothing(img,sigma,ws):

    ks = int(ws/2)
    h = np.zeros((ws,ws))
    for i in range(ws):
        for j in range(ws):
            h[i,j] = np.exp(-((i)**2+(j)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    sum_h = np.sum(h)
    h = h/sum_h

    img_gs = np.zeros(img.shape)
    img = np.pad(img, ks)
    m,n = img.shape

    for i in range(ws,m-ws):
        for j in range(ws,n-ws):
            img_gs[i, j] = np.sum(img[i-ks:i+ks+1, j-ks:j+ks+1]*h)

    return img_gs


def sobel(img):
   
    ws = 3
    ks = ws//2

    #kernel
    Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    img_s1 = np.zeros(img.shape)
    img_s2 = np.zeros(img.shape)
    img_p = np.pad(img, (ks,ks), constant_values=1)
    m,n = img_p.shape
 
    for i in range(ws,m-ws):
        for j in range(ws,n-ws):

            img_s1[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gx)
            img_s2[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gy)
             
    #gradient magnitude
    img_n = np.sqrt(img_s1**2 + img_s2**2)

    return img_n


def perwitt(img):

    ws = 3
    ks = ws//2

    #kernel
    Gx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    Gy = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])

    img_s1 = np.zeros(img.shape)
    img_s2 = np.zeros(img.shape)
    img_p = np.pad(img, ks)
    m,n = img_p.shape

    for i in range(ws,m-ws):
        for j in range(ws,n-ws):

            img_s1[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gx)
            img_s2[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Gy)
             
    #gradient magnitude       
     
    img_n = np.sqrt(img_s1**2 + img_s2**2) 

    return img_n


#1st order Laplacian
def f_laplacian(img):
   
    ws = 3
    ks = int(ws/2)

    #kernel
    Lp = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])/8

    img_s1 = np.zeros(img.shape)
    img_p = np.pad(img, ks)
    m,n = img_p.shape

    for i in range(ws,m-ws):
        for j in range(ws,n-ws):
            img_s1[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Lp)

    return img_s1

#2nd order Laplacian
def s_laplacian(img):
   
    ws = 3
    ks = int(ws/2)

    #kernel
    Lp = np.array([[1,1,1],[1, -8, 1],[1, 1, 1]])/16

    img_s1 = np.zeros(img.shape)
    img_p = np.pad(img, ks)
    m,n = img_p.shape

    for i in range(ws,m-ws):
        for j in range(ws,n-ws):
            img_s1[i, j] = np.sum(img_p[i-ks:i+ks+1, j-ks:j+ks+1]*Lp)

    return img_s1


#zero crossing...
def zero_crossing(img):
    m,n = img.shape
    zc_img = np.zeros((m,n))

    # Check the sign (negative or positive) of all the pixels around each pixel
    for i in range(1,m-1):
        for j in range(1,n-1):
            neg_count = 0
            pos_count = 0
            for a in range(-1, 2):
                for b in range(-1,2):
                    if(img[i+a,j+b] < 0):
                        neg_count += 1
                    elif(img[i+a,j+b] > 0):
                        pos_count += 1

            z_c = ( (neg_count > 0) and (pos_count > 0) )
            if(z_c):
                zc_img[i,j] = 255

    return zc_img


def rotat(img, k):
    return tr.rotate(img, k)



img_g1 = g_smoothing(img1,sigma,ws)
img_g2 = g_smoothing(img2,sigma,ws)
img_g3 = g_smoothing(img3,sigma,ws)
img_g4 = g_smoothing(img4,sigma,ws)

f_ls = zero_crossing(f_laplacian(img1))
s_ls = zero_crossing(s_laplacian(img1))

f_ls1 = zero_crossing(f_laplacian(img3))
s_ls1 = zero_crossing(s_laplacian(img3))

fig1, (ax1, ax2) = plt.subplots(2,2)
ax1[0].imshow(img_g1, cmap='gray')
ax1[0].set_title("Checkerboard")
ax1[0].axis('off')

ax1[1].imshow(img_g2, cmap='gray')
ax1[1].set_title("NoisyCheckerboard")
ax1[1].axis('off')

ax2[0].imshow(img_g3, cmap='gray')
ax2[0].set_title("Coins")  
ax2[0].axis('off')

ax2[1].imshow(img_g4, cmap='gray')
ax2[1].set_title("NoisyCoins")  
ax2[1].axis('off')

plt.show()


fig2, (ax1, ax2) = plt.subplots(2,2)
ax1[0].imshow(sobel(rotat(img2, 45)), cmap='gray')
ax1[0].set_title("NoisyCheckerboard")
ax1[0].axis('off')

ax1[1].imshow(sobel(rotat(img_g2, 45)), cmap='gray')
ax1[1].set_title("guss + NoisyCheckerboard")
ax1[1].axis('off')

ax2[0].imshow(perwitt(rotat(img2, 45)), cmap='gray')
ax2[0].set_title("NoisyCheckerboard")  
ax2[0].axis('off')

ax2[1].imshow(perwitt(rotat(img_g2, 45)), cmap='gray')
ax2[1].set_title("guss + NoisyCheckerboard")  
ax2[1].axis('off')

plt.show()

fig3, (ax1, ax2) = plt.subplots(2,2)
ax1[0].imshow(sobel(img4), cmap='gray')
ax1[0].set_title("NoisyCoins")
ax1[0].axis('off')

ax1[1].imshow(sobel(img_g4), cmap='gray')
ax1[1].set_title("guss + NoisyCoins")
ax1[1].axis('off')

ax2[0].imshow(perwitt(img4), cmap='gray')
ax2[0].set_title("NoisyCoins")  
ax2[0].axis('off')

ax2[1].imshow(perwitt(img_g4), cmap='gray')
ax2[1].set_title("guss + NoisyCoins")  
ax2[1].axis('off')

plt.show()

fig4, (ax1, ax2) = plt.subplots(2,2)
ax1[0].imshow(f_ls, cmap='gray')
ax1[0].set_title("l1")
ax1[0].axis('off')

ax1[1].imshow(s_ls, cmap='gray')
ax1[1].set_title("l2")
ax1[1].axis('off')

ax2[0].imshow(f_ls1, cmap='gray')
ax2[0].set_title("l1")  
ax2[0].axis('off')

ax2[1].imshow(s_ls1, cmap='gray')
ax2[1].set_title("l2")  
ax2[1].axis('off')

plt.show()

 

