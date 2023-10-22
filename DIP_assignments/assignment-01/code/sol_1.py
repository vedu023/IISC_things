import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def hist(path):
    img = np.array(io.imread(path))
    h,w = img.shape
    hist_f = np.zeros(256)
    #hist_g = {i : 0 for i in range(256)}

    for i in range(h):
        for j in range(w):
            for k in range(256):
                if(img[i][j] == k):
                    hist_f[k] = int(hist_f[k] + 1)
                    break
    #hist_f =  hist_g.values()
    return hist_f

path = 'img/GulmoharMarg.png'
img = io.imread(path)
hist_w = hist(path)

print('=='*50)
print('\nhistogram without lib function...(first 10 values)\n')
print(hist_w[:10])

print('=='*50)
print('\nhistogram without lib function...(first 10 values)\n')
hist_l, _ = np.histogram(img, bins = 256)
print(hist_l[:10])

print('\n both are equal...?')
print('true' if hist_l.all() == hist_w.all() else 'false')

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(hist_w)
ax1.set_title("without lib function")

ax2.plot(hist_l)
ax2.set_title("lib function")

plt.show()