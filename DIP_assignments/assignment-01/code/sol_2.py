import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def Otsu_algo(img):

    hist, b = np.histogram(img, bins=256) # histogram
    
    b = b[:-1] 
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]

    # Get the class means 
    mu0 = np.cumsum(hist * b) / w0
    mu1 = (np.cumsum((hist * b)[::-1]) / w1[::-1])[::-1]

    # Get the class variance
    var0 = np.cumsum(hist* (b - mu0)**2)/w0
    var1 = (np.cumsum((hist* (b - mu1)**2)[::-1])/w1[::-1])[::-1]

    s_b = w0 * w1 * (mu0 - mu1) ** 2  # between class variance
    s_w = w0 * var0 + w1 * var1   # within class variance

    # final thresholding
    max_v = np.argmax(s_b)
    min_v = np.argmin(s_w)
    t1 = b[max_v]
    t2 = b[min_v]

    _, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(s_b)
    ax1.set_title("between class variance")

    ax2.plot(s_w)
    ax2.set_title("within class variance")

    return t1, t2

path = 'img/coins.png'
img = io.imread(path)

# result
t1, t2 = Otsu_algo(img)
print('='*50)
print(f'threshold by between class : {t1}')
print(f'threshold by within class : {t2}')

print('both are equal..?')
print('true' if t1 == t2 else 'false')

# binary image
img[img<t1] = 0
img[img>=t1] = 1

_, ax = plt.subplots()
ax.imshow(img, cmap = 'gray')
ax.set_title('binary image')
plt.show()