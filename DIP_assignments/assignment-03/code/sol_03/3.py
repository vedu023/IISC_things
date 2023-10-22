import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


# input 
img = io.imread('Images/characters.tif')
m,n = img.shape

def dft_matrix(img):

    N = img.shape[0]  
    w = np.exp((-2 * np.pi * 1j) / N)  
    r = np.arange(N)
    w_matrix = np.vander(w ** r, increasing=True)  
    return w_matrix

def mse(x,y):
    me = ((x - y)**2).mean(axis = 1) 
    return np.sum(me)


def fscs(img):
    img = np.array(img)
    k = 256

    A = np.min(img)
    B = np.max(img)
    N = B-A

    img_f =  ((k-1)/N) * (img - A)
     
    return img_f

# to compute A matrix
def dft_vls(dft):
    m_spe = np.log(abs(dft))
    img = fscs(m_spe)
    
    plt.plot(img)
    plt.show()


A = dft_matrix(img)

F1 = A @ img @ A.T  # F = AfA_t
F2 = np.fft.fft2(img)  # lib function

dft_vls(F1)
dft_vls(F2)

img_F1 = np.fft.ifft2(F1)
img_F2 = np.fft.ifft2(F2)

img_F1 = np.uint8(np.real(img_F1))
img_F2 = np.uint8(np.real(img_F2))

print(mse(img_F2, img_F1))

fig2, (a1,a2) = plt.subplots(1,2)
a1.imshow(img_F1, cmap = 'gray')
a1.set_title('Matrix Method')

a2.imshow(img_F2, cmap = 'gray')
a2.set_title('Library function')
plt.show()