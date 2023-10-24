   
import cv2
import numpy as np
import matplotlib.pyplot as plt
 

def modified_imgs(img):
    h, w, _ = img.shape
    k = 2         # scaling factor
    
    up_img = cv2.resize(img, (h*k, w*k) )
    down_img = cv2.resize(img, (h//k, w//k) )
    ro_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    blurred_img = cv2.blur(img, (3,3))
    
    noise = np.random.randn(h, w, 3)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return [img, up_img, down_img, ro_img, blurred_img, noisy_img]
 
    
if __name__ == "__main__":
        
    sift = cv2.SIFT_create()
    
    img1 = cv2.imread('AIP_assignments/a1/first.jpeg')
    img2 = cv2.imread('AIP_assignments/a1/second.jpeg')
     
    imgs = [modified_imgs(img1), modified_imgs(img2)]
    
    for i in imgs:
        for j in i:
            keypoints, _ = sift.detectAndCompute(j, None)
            img_with_kps = cv2.drawKeypoints(j, keypoints, None)
            plt.imshow(img_with_kps)
            plt.axis('off')
            plt.show()
        