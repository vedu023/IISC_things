import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

#input
img = io.imread('Images/characters.tif')
m,n = img.shape 
shape = m,n

D0 = 100    #cutoff freq

F = np.fft.fftshift(np.fft.fft2(img))  #fft

# distance function
def dist(u,v):
    return np.sqrt((u-m/2)**2+(v-n/2)**2)

# ideal low pass filter
def ILPF(m,n, D0):
    h = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if(dist(i,j) <= D0):
                h[i,j] = 1
            else:
                h[i,j] = 0
            
    return h

# guassain low pass filter
def GLPF(m, n, D0):
    h = np.zeros((m,n))
    for u in range(m):
        for v in range(n):
            h[u,v] =  np.exp(-(dist(u,v)**2)/(2*(D0**2)))

    return h

h1 = ILPF(m,n,D0)
h2 = GLPF(m,n,D0)

img1 = F*h1
img2 = F*h2

# inv dft
img1_s = np.fft.ifft2(np.fft.fftshift(img1))
img2_s = np.fft.ifft2(np.fft.fftshift(img2))
 
img1_b = np.uint8(np.real(img1_s))
img2_b = np.uint8(np.real(img2_s))

# plot..
fig, (a1,a2) = plt.subplots(1,2)
a1.imshow(img1_b, cmap = 'gray')
a1.set_title('ILPF')

a2.imshow(img2_b, cmap = 'gray')
a2.set_title('GLPF')
plt.show()