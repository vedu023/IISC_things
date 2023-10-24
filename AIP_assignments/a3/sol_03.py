import skimage.io
import skimage.color
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
from sol_02 import LPF, MSE

# High Pass filter
def high_pass(a, b):
    matrix = [-1,-1,-1,-1,8,-1,-1,-1,-1]
    matrix = np.reshape(matrix,(3,3))
    return matrix[a+1][b+1]

# Function to perform the image sharpening
def sharpen(blurred_img, lamda):
    kernel = 3
    std_dev = 1
    h,width = blurred_img.shape
    
    mu_y = LPF(blurred_img,kernel,std_dev)
    
    m,n = blurred_img.shape
      
    padded_blurimage=np.pad(blurred_img,((1,1),(1,1)))
      
    image_highpass=np.zeros((m,n))
    n1=0;n2=0
    for n1 in range(0,3):
      for n2 in range(0,3):
        image_highpass = image_highpass + (padded_blurimage[n1:n1+m,n2:n2+n])*high_pass((n1-1),(n2-1))
        n2=n2+1
      n1=n1+1
    
    out_img = mu_y + lamda*image_highpass 
    out_img = np.clip(out_img,0,255)
    return out_img

if __name__ == "__main__":    

    input_img = skimage.io.imread('lighthouse2.bmp')
    input_img = ((rgb2gray(input_img))*255).astype('uint8')
    
    mean = 0.0 
    var = 100.0   
    noisy_img = input_img + np.random.normal(mean, var**(0.5) , input_img.shape)
    noisy_img = np.clip(noisy_img, 0, 255) 
    
    denoised = LPF(noisy_img,kernel=3,std_dev =1)    
    mse = np.inf
    lamb = 0
    for lamda1 in range(0,30):
      lamda = lamda1/10
      img_sharp = sharpen(denoised,lamda)
      particular = MSE(img_sharp,input_img)
      if(particular < mse):
        mse = particular
        lamb = lamda
    
    img_sharp = sharpen(denoised, lamb)
    plt.figure()
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(img_sharp, cmap='gray')
    plt.title("Sharpened Image")
    plt.imsave("sharpened.jpg", img_sharp, cmap="gray")
    
    plt.subplot(122)
    plt.axis("off")
    plt.imshow(input_img, cmap='gray')
    plt.title("Input Image")
    
    print("\nGain the minimizes MSE:", lamb," and Mean Squared Error:", MSE(img_sharp,input_img))
    
    for lamd in [1, 2, 3, 4]:
        img_sharp = sharpen(denoised, lamd)
        plt.figure()
        plt.axis("off")
        plt.imshow(img_sharp, cmap='gray')
        plt.title("Lambda: %d" %lamd)
        plt.imsave(f"{lamd}_sharpened.jpg", img_sharp, cmap="gray")