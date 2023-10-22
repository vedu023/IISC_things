import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io


# input
img = io.imread('Images/noisybook.png')

M = 10
fs = 2*M + 1
a = 178
b = 260
s = 5


def guss2(s,x,y):
    return (np.exp(-((x)**2 + (y)**2)/(2* s**2)))/(2*np.pi*s**2)

def guss(s,x):
    return (np.exp(-((x)**2)/(2* s**2)))/(2*np.pi*s**2)

# G_map...
def g_flt(s, fs):

    ks = fs//2
    g_filter = np.zeros((fs,fs))
    g_map = np.zeros((fs,fs))
 
    for x in range(fs):
        for y in range(fs):
            g_filter[x,y] = guss2(s, x-ks, y-ks)
    g_map = g_filter/np.sum(g_filter)
     
    return g_map

# H_map
def h_flt(img, s, a, b, fs):

    ks = fs//2
    img = np.pad(img, ks, mode = 'constant', constant_values=1)
    img = np.int32(img)

    b_flt = np.zeros((fs,fs))
    h_map = np.zeros((fs,fs))

    for i in range(fs):
        for j in range(fs):
            b_flt[i,j] = guss(s, img[a,b]-img[a+i-ks, j+b-ks])
    h_map = b_flt/(np.sum(b_flt))

    return h_map

g_map = g_flt(s, fs)
h_map = h_flt(img, s, a, b, fs)

f = g_map*h_map

fig, (a1,a2, a3) = plt.subplots(1,3)
a1.plot(g_map)
a1.set_title('g_map')

a2.plot(h_map)
a2.set_title('h_map')

a3.plot(f)
a3.set_title('g*h')
plt.show()
