from PIL import Image
import matplotlib.pyplot as plt

setName = input("set name: ")
path = f"dataset/{setName}/"

# 이미지 경로가 담긴 텍스트 파일 읽기
with open(f"{setName}.log", "r") as file:
    lines = file.readlines()

for line in lines:
	file1, file2, *_ = line.split()

	# 이미지 읽기
	image1 = Image.open(path+file1)
	image2 = Image.open(path+file2)

	# 두 이미지를 함께 표시
	plt.figure(figsize=(10, 5))

	# 첫 번째 이미지
	plt.subplot(1, 2, 1)
	plt.imshow(image1)
	plt.title(file1)
	plt.axis('off')

	# 두 번째 이미지
	plt.subplot(1, 2, 2)
	plt.imshow(image2)
	plt.title(file2)
	plt.axis('off')

	# 화면에 표시
	plt.show()

	yes = input(f"Delete {file2}? ")
	if yes == 'y':
		os.remove(path+file2)
		print(path+file2, "is removed.")
