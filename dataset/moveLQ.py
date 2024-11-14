import sys
import os
import cv2
import shutil
import numpy as np
from PIL import Image
from collections import Counter

# 원본 폴더와 임시 폴더 경로 설정
if len(sys.argv) != 3: quit()
source_folder = sys.argv[1]
temp_folder = sys.argv[2]

# 임시 폴더가 없으면 생성
os.makedirs(temp_folder, exist_ok=True)

def is_blurry(image, threshold=90):
	"""이미지가 흐릿한지 여부를 결정"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
	return laplacian_var < threshold

def is_color_biased(image, threshold=0.4):
	"""이미지 색상이 특정 색으로 편중되었는지 확인"""
	pixels = np.array(image).reshape(-1, 3)
	colors, counts = np.unique(pixels, axis=0, return_counts=True)
	max_color_ratio = counts.max() / counts.sum()
	return max_color_ratio > threshold

def process_images(source_folder, temp_folder):
	"""이미지를 분석하고, 흐릿하거나 색 편중된 이미지를 임시 폴더로 이동"""
	for filename in os.listdir(source_folder):
		if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
			img_path = os.path.join(source_folder, filename)
			image = cv2.imread(img_path)
			pil_image = Image.open(img_path)
			if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')


			# 화질 저하 확인
			blurry = is_blurry(image)

			# 색상 편중 확인
			color_biased = is_color_biased(pil_image)

			# 조건에 맞는 이미지를 임시 폴더로 이동
			if blurry or color_biased:
				shutil.move(img_path, os.path.join(temp_folder, filename))
				print(f"move {filename} to temporary folder.")

process_images(source_folder, temp_folder)
