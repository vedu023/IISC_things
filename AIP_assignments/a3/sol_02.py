import skimage.io
import skimage.color
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Low Pass Filter to denoise image
def LPF(image, kernel, std_dev):

    h,w = image.shape
    matrix_img = np.zeros((kernel**2,h,w))
    for j in range(kernel):
      for i in range(kernel):
        translation_matrix = np.float32([ [1,0,(i - (kernel-1)/2)], [0,1,(j - (kernel-1)/2)] ])
        matrix_img[(j*kernel + i),:,:] = cv2.warpAffine(image, translation_matrix, (w,h))
    
    gaussian_out = np.zeros((h,w))
    multiplier = 0
    for j in range(kernel):
      for i in range(kernel):
        multiplier = multiplier + (1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2))
        gaussian_out = gaussian_out + (1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2))*matrix_img[(j*kernel + i),:,:]
    gaussian_out = gaussian_out/multiplier
    gaussian_out = np.clip(gaussian_out, 0, 255)
    return gaussian_out 

# To find MSE between img1 and img2
def MSE(image1, image2):
    error = (image1 - image2)**2
    mse = np.mean(error)
    return mse

# Part b) 
def MMSE(noisy_image,sigma2_z):
    kernel = 3
    std_dev = 1
    
    h,width = noisy_image.shape
    
    mu_y = LPF(noisy_image, kernel, std_dev)
    y1 = (noisy_image - mu_y)
    
    multiplier = 0
    for j in range(kernel):
      for i in range(kernel):
        multiplier = multiplier + (1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2))
    
    w_sum = 0
    for j in range(kernel):
      for i in range(kernel):
        w = multiplier*((1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2)))
        if((i - (kernel-1)/2)==0 and (j - (kernel-1)/2) == 0):
          w_sum = w_sum + (1-w)**2
        else:
          w_sum = w_sum + (w)**2
    
    sigma_z1_2 = w_sum*sigma2_z
    
    sigma_y1_2 = np.var((y1))
    sigma_x1_2 = sigma_y1_2 - sigma_z1_2
    
    out_img = mu_y + (sigma_x1_2/sigma_y1_2)*y1
    out_img = np.clip(out_img,0,255)
    return out_img
  
# Part c)
def MMSE_adaptive(noisy_image,sigma2_z):
    kernel = 3
    std_dev = 1
    
    h,width = noisy_image.shape
    
    mu_y = LPF(noisy_image, kernel, std_dev)
    y1 = (noisy_image - mu_y)
    
    multiplier = 0
    for j in range(kernel):
      for i in range(kernel):
        multiplier = multiplier + (1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2))
    
    w_sum = 0
    for j in range(kernel):
      for i in range(kernel):
        w = multiplier*((1/(2*np.pi*(std_dev**2)))*np.exp(-1*((i - (kernel-1)/2)**2 + (j - (kernel-1)/2)**2)/(2*std_dev**2)))
        if((i - (kernel-1)/2)==0 and (j - (kernel-1)/2) == 0):
          w_sum = w_sum + (1-w)**2
        else:
          w_sum = w_sum + (w)**2
    
    sigma_z1_2 = w_sum*sigma2_z
    
    out_img = np.zeros((np.shape(mu_y)))
    weight = np.zeros((np.shape(mu_y)))
    
    for i in range(0,h,6):
      for j in range(0,width,6):
        sigma_y1_2 = np.var((y1[i:i+11,j:j+11]))
        sigma_x1_2 = sigma_y1_2 - sigma_z1_2
        out_img[i:i+11,j:j+11] = out_img[i:i+11,j:j+11] + (sigma_x1_2/sigma_y1_2)*y1[i:i+11,j:j+11] 
        weight[i:i+11,j:j+11]+=1
    
    weight[weight==0]=1
    out_img = out_img/weight
    
    out_img = mu_y + out_img
    out_img = np.clip(out_img,0,255)
    return out_img

if __name__ == "__main__":
    
    input_img = skimage.io.imread('lighthouse2.bmp')
    input_img = ((rgb2gray(input_img))*255).astype('uint8')
    
    mean = 0.0 
    var = 100.0   
    noisy_img = input_img + np.random.normal(mean, var**(0.5) , input_img.shape)
    noisy_img = np.clip(noisy_img, 0, 255) 
    
    best_kernel = 0
    best_std = 0
    mse = np.inf
    
    for kernel in [3, 7, 11]:
        for std in [0.1, 1, 2, 4, 8]:
            img_gauss = LPF(noisy_img, kernel, std)
            temp = MSE(img_gauss, input_img)
            if (temp < mse):
                best_kernel = kernel
                best_std = std
                mse = temp

    print("For the Low Pass Filter Kernel\nBest Kernel Size:", best_kernel, "and Std. deviation:", best_std)
    img_gauss = LPF(noisy_img, kernel=best_kernel, std_dev=best_std)
    print("\nMean Squared Error between Low Pass filtered image and Input Image: ", MSE(img_gauss, input_img))
       
    img_mmse = MMSE(noisy_img, 100)
    print("Mean Squared Error for Global MMSE: ", MSE(img_mmse, input_img))
    
    img_mmse_adaptive = MMSE_adaptive(noisy_img,100)
    print("Mean Squared Error for Adaptive MMSE: ", MSE(img_mmse_adaptive, input_img))
    
    out_img_gauss = np.clip(img_gauss,0,255).astype('uint8')
    out_img_mmse = np.clip(img_mmse,0,255).astype('uint8')
    out_img_adaptive = np.clip(img_mmse_adaptive,0,255).astype('uint8')
    
    plt.figure()
    plt.subplot(221)
    plt.axis("off")
    plt.imshow(input_img, cmap ="gray")
    plt.title("Input Image")
    plt.imsave("input_img.jpg", input_img, cmap="gray")
    
    plt.subplot(222)
    plt.axis("off")
    plt.imshow(out_img_gauss, cmap ="gray")
    plt.title("Low Pass Filtered")
    plt.imsave("lpf.jpg", out_img_gauss, cmap="gray")
    
    plt.subplot(223)
    plt.axis("off")
    plt.imshow(out_img_mmse, cmap ="gray")
    plt.title("MMSE Filtered")
    plt.imsave("mmse.jpg", out_img_mmse, cmap="gray")
    
    plt.subplot(224)
    plt.axis("off")
    plt.imshow(out_img_adaptive, cmap ="gray")
    plt.title("Adaptive MMSE Filtered")
    plt.imsave("adap_mmse.jpg", out_img_adaptive, cmap="gray")