import numpy as np
import skimage.io as io

path = 'img/DoubleColorShapes.png'
img = io.imread(path)
h,w = img.shape

reg = [] 

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

  elemnets = elemnets[freq>100]              
  elemnets = elemnets[elemnets>0]

  return len(elemnets)


for t in range(1,255):

    img[img<t] = 1
    img[img>=t] = 0

    img = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=1)

    reg.append(cc(img))

mser, t = np.max(reg), np.argmax(reg)
print(mser)


