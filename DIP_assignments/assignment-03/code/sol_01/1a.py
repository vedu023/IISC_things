
import numpy as np
import matplotlib.pyplot as plt

M = 501
N = 501

# full scale contrast stretching
def fscs(img):
    img = np.array(img)
    k = 256

    A = np.min(img)
    B = np.max(img)
    N = B-A
    img_f =  ((k-1)/N) * (img - A)
    
    return img_f

# for visulaisation of sinusoidal images
def sin_ft(shape, u, v):

    M,N = shape
    s_img = np.zeros((M,N))

    for m in range(M):
        for n in range(N):
            s_img[m,n] = np.sin(2*np.pi*((u*m/M) + (v*n/N))) # sinusoidal 

    dft = np.fft.fft2(s_img)
    dft_s = np.fft.fftshift(dft)
    m_spe = np.log(abs(dft_s))    # magnitude spectrum  
    img = fscs(m_spe)

    # plot 
    fig, (a1,a2) = plt.subplots(1,2)
    a1.imshow(img, cmap = 'gray')
    a2.plot(img)
    plt.show()

    return s_img, dft_s

img1 = sin_ft((M,N), 40, 60)
img2 = sin_ft((M,N), 20, 100)

img_0 = img1[0] + img2[0]  #sum of orignal images
img_d = img1[1] + img2[1]  #sum of dfts

# inv dft
img_i = np.fft.ifft2(np.fft.fftshift(img_d))
img = np.uint8(np.real(img_i))

# final output plot 
fig, (a1,a2) = plt.subplots(1,2)
a1.imshow(img_0, cmap = 'gray')
a1.set_title('Original sum of imgs')

a2.imshow(img, cmap = 'gray')
a2.set_title('iDFT of sum')
plt.show()