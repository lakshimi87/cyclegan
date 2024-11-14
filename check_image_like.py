import cv2
import os

PathFormat = "dataset/%s/"

def isLike(image1, image2):
	hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
	hist_img1[255, 255, 255] = 0 #ignore all white pixels
	cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
	hist_img2[255, 255, 255] = 0  #ignore all white pixels
	cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	# Find the metric value
	metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
	return round(metric_val, 2) > 0.8

setName = input("set name: ")
path = PathFormat%setName
files = os.listdir(path)
files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

for i in range(len(files)-1):
	image1 = cv2.imread(path+files[i])
	for j in range(i+1, len(files)):
		image2 = cv2.imread(path+files[j])
		if isLike(image1, image2): print(files[i], files[j])
