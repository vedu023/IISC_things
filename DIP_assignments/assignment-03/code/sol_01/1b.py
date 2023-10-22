import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

#input
img = io.imread('Images/dynamicSine.png')
m,n = img.shape 
shape = m,n

D0 = 100   # cutoff freq 

F = np.fft.fftshift(np.fft.fft2(img))  # Fast Fourier Transform

# istance function
def dist(u,v):
    return np.sqrt((u-m/2)**2+(v-n/2)**2)

# ideal low pass filter
def ILPF(D0):
    h = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if(dist(i,j) <= D0):
                h[i,j] = 1
            else:
                h[i,j] = 0
            
    return h

# ideal high pass filter
def IHPF(D0):
    return 1 - ILPF(D0)

# ideal band pass filter
def IBPF(D1, D2):
    return IHPF(D1)*ILPF(D2)
            

h1 = ILPF(20)
h2 = IHPF(60)
h3 = IBPF(20,40)
h4 = IBPF(40,60)

img1 = F*h1
img2 = F*h2
img3 = F*h3
img4 = F*h4

# inv dft
img1_s = np.fft.ifft2(np.fft.fftshift(img1))
img2_s = np.fft.ifft2(np.fft.fftshift(img2))
img3_s = np.fft.ifft2(np.fft.fftshift(img3))
img4_s = np.fft.ifft2(np.fft.fftshift(img4))
 
img1_b = np.uint8(np.real(img1_s))
img2_b = np.uint8(np.real(img2_s))
img3_b = np.uint8(np.real(img3_s))
img4_b = np.uint8(np.real(img4_s))

# ploting
fig1, (a1,a2) = plt.subplots(2,2)
a1[0].plot(h1)
a1[0].axis('off')
a1[0].set_title('ILPF')

a1[1].plot(h2)
a1[1].axis('off')
a1[1].set_title('IHPF')

a2[0].plot(h3)
a2[0].axis('off')
a2[0].set_title('IBPF-1')

a2[1].plot(h4)
a2[1].axis('off')
a2[1].set_title('IBPF-2')

fig2, (a1,a2) = plt.subplots(2,2)
a1[0].imshow(img1_b, cmap = 'gray')
a1[0].axis('off')
a1[0].set_title('ILPF')

a1[1].imshow(img2_b, cmap = 'gray')
a1[1].axis('off')
a1[1].set_title('IHPF')

a2[0].imshow(img3_b, cmap = 'gray')
a2[0].axis('off')
a2[0].set_title('IBPF-1')

a2[1].imshow(img4_b, cmap = 'gray')
a2[1].axis('off')
a2[1].set_title('IBPF-2')
plt.show()