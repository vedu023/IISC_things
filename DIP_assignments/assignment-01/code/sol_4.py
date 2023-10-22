import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

path = 'img/Shapes.png'
img = io.imread(path)
h,w = img.shape

# binary image...
def threshold(img):
    
    hist, b = np.histogram(img, bins=256) # histogram
    
    b = b[:-1] 
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
 
    mu0 = np.cumsum(hist * b) / w0
    mu1 = (np.cumsum((hist * b)[::-1]) / w1[::-1])[::-1]

    s_b = w0 * w1 * (mu0 - mu1) ** 2  
    max_v = np.argmax(s_b)
    t = b[max_v]
    
    return t

t = threshold(img)

img[img<t] = 0
img[img>=t] = 1

# padding...
img = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=1)

# connected component...
def cc(img):

  r = np.zeros((h,w))
  k = 1
  for i in range(1,h+1):
    for j in range(1,w+1):

      if img[i,j] == 1:

        if ((img[i,j-1] == 0) and (img[i-1,j] == 0)):
          r[i,j] = k
          k += 1

        elif ((img[i,j-1] == 1) and (img[i-1,j] == 0)):
          r[i,j] = r[i,j-1]
        
        elif ((img[i,j-1] == 0) and (img[i-1,j] == 1)):
          r[i,j] = r[i-1,j]

        elif ((img[i,j-1] == 1) and (img[i-1,j] == 1)):
          r[i,j] = r[i-1,j]
          if r[i,j-1]!=r[i-1,j]:
                r[r == r[i,j-1]] = r[i-1,j]

  elemnets,freq=np.unique(r,return_counts=True)

  elemnets = elemnets[freq>100]               # cal no. of elements...
  circles = elemnets
  freq = freq[freq>100]
  elemnets = elemnets[elemnets>0]

  circles = circles[freq<600]                 # cal no. of circle...
  circles = circles[circles>0]

  return len(elemnets), len(circles)

elements, circles = cc(img)

print('='*50)
print(f'\nno of elements:: {elements}')
print(f'no of circles:: {circles}')
print('='*50)