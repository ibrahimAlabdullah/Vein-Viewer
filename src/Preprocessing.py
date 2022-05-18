import numpy as np
import cv2
import matplotlib.pyplot as plt  

def adjust_gamma(image, gamma=1.0):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(image,gamma_table)

def creatMask(img,roi=50):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[roi:-roi,roi:-roi] = 1
    #plot(mask)
    img = cv2.bitwise_and(img, img, mask=mask) 
    return img

def Remove_hair(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    Clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    ClaheImage = Clahe.apply(image) 
    #plot(ClaheImage)
    blurImg =cv2.GaussianBlur(ClaheImage,(25,25),0)
    #plot(blurImg)
    return blurImg 

