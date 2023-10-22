import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

#input
img = io.imread('Images/noisybook.png')
m,n = img.shape
s = 5
f_s = 7  


def guss2(s,x,y):
    return (np.exp(-((x)**2 + (y)**2)/(2* s**2)))/(2*np.pi*s**2)

# Gaussian filter...
def g_flt(s, fs):

    ks = fs//2
    g_filter = np.zeros((fs,fs)) 
    for x in range(fs):
        for y in range(fs):
            g_filter[x,y] = guss2(s, x-ks, y-ks)
     
    return g_filter

# Gaussian smoothing...
def g_smoothing(img, s, fs):

    img = np.pad(img, fs//2, mode = 'constant', constant_values=1) 
    img_g = np.zeros((fs,fs))
    img_gsmooth = np.zeros((m,n))
    
    gflt = g_flt(s, fs)
    ws = np.sum(gflt)

    for i in range(m):
        for j in range(n):

            for k1 in range(fs):
                for k2 in range(fs):
                    img_g[k1,k2] = gflt[k1,k2]*img[i+k1,j+k2]    
            
            img_gsmooth[i-fs,j-fs] = np.sum(img_g)/ws
    
    return img_gsmooth


# bilitral smoothing...
def blt_smoothing(img, s, fs):

    ks = fs//2
    b_flt = np.zeros((fs,fs))
    win = np.zeros((fs,fs))
    img_bilateral = np.zeros((m,n))
    img = np.pad(img, ks, mode = 'constant', constant_values=1)
    img = np.array(img, dtype=np.int32)

    for i in range(m):
        for j in range(n):
            
            #bilateral filter
            for x in range(fs):
                for y in range(fs):
                    x1 = guss2(s, x-ks, y-ks)
                    x2 = np.exp(-((img[i,j] - img[x,y])**2)/(np.sqrt(2*np.pi)*s**2))
                    b_flt[x,y] = x1*x2
            wb = np.sum(b_flt)
        
            for k1 in range(fs):
                for k2 in range(fs):
                    win[k1,k2] = b_flt[k1,k2]*img[i+k1,j+k2]    
            
            img_bilateral[i-ks,j-ks] = np.sum(win)/wb
    
    return img_bilateral


img_g = g_smoothing(img, s, f_s)
img_b = blt_smoothing(img, s, f_s)

# plot
fig, (a1,a2) = plt.subplots(1,2)
a1.imshow(img_g, cmap = 'gray')
a1.set_title('g_smoothing')

a2.imshow(img_b, cmap = 'gray')
a2.set_title('blt_smoothing')
plt.show()




