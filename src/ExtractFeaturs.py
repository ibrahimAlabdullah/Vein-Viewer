
import numpy as np
import cv2
import matplotlib.pyplot as plt  
from Preprocessing import adjust_gamma, creatMask, Remove_hair

def extractFeature(Remove_hairImg):
    Clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    op = cv2.morphologyEx(Remove_hairImg, cv2.MORPH_OPEN, kernel)
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    op = cv2.morphologyEx(cl, cv2.MORPH_OPEN, kernel)
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)
    MorphologyImage = Clahe.apply(cv2.subtract(cl, Remove_hairImg))    
    #plot(MorphologyImage)
    return MorphologyImage
    
def NoiseSolving(image): 
    mask = np.ones(image.shape , dtype="uint8") 
    __ ,solveNoise = cv2.threshold(image,20,255,cv2.THRESH_BINARY)	
    #plot(solveNoise)
    contours, _ = cv2.findContours(solveNoise,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sort_contours=[]
    for c in contours: 
        sort_contours.append(cv2.contourArea(c))
    sort_contours.sort()
    area=sum(sort_contours)/20
    big_contours=[]
    for c in contours: 
        if cv2.contourArea(c) > area:   
            big_contours.append(c[int(len(c)/4)][0])
            cv2.drawContours(mask, [c], -1, 0, -1)	  
    #plot(mask)
    return mask,big_contours

def drawFeatures(mask,orgImage):
    blood_vessels = cv2.bitwise_not(mask) 
    orgImage = cv2.bitwise_and(orgImage,blood_vessels, mask=mask)
    #orgImage=orgImage[50:200,50:200]
    return orgImage

def getCoordinates(big_contours,BloodVessels):
    for i in range(len(big_contours)):
        cv2.putText(BloodVessels, ".", (big_contours[i][0],big_contours[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 10, cv2.LINE_AA)
    return BloodVessels

def VeinDetection(image,grayImg):
    Remove_hairImg = Remove_hair(image)
    extractFeatureImg = extractFeature(Remove_hairImg)
    mask,big_contours = NoiseSolving(extractFeatureImg) 
    BloodVessels = drawFeatures(mask,grayImg) 
    #BloodVessels = getCoordinates(big_contours,BloodVessels) 
    return BloodVessels


