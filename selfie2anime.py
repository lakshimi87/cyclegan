"""
Title: selfie to anime
Description: selfie to anime with GAN
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import itertools
import datetime
import time
from torchvision.utils import save_image, make_grid
from torchvision import datasets

# H/W acceleration
if torch.cuda.is_available():
	device = torch.device('cuda')
	torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
	device = torch.device('mps')
else:
	device = torch.device('cpu')

def loadImage(fileName):
	image = Image.open(fileName)
	if image.mode != 'RGB': image = image.convert('RGB')
	return image

# image dataset class
class ImageDataset(Dataset):
	def __init__(self, root, trans, unaligned=True, mode='train'):
		self.transform = transforms.Compose(trans)
		self.unaligned = unaligned
		self.filesA = sorted(glob.glob(os.path.join(root, f'{mode}A')+'/*.*'))
		self.filesB = sorted(glob.glob(os.path.join(root, f'{mode}B')+'/*.*'))
		self.lenA = len(self.filesA)
		self.lenB = len(self.filesB)

	def __getitem__(self, index):
		idx = index%self.lenA
		imageA = loadImage(self.filesA[idx])
		idx = random.randrange(self.lenB) if self.unaligned else index%self.lenB
		imageB = loadImage(self.filesB[idx])
		itemA = self.transform(imageA)
		itemB = self.transform(imageB)
		return { 'A':itemA, 'B':itemB }

	def __len__(self):
		return self.lenA

# initialize weights by normal
def initWeightsNormal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		if hasattr(m, 'bias') and m.bias != None:
			nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm2d') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0.0)

# ResidualBlock class
class ResidualBlock(nn.Module):
	def __init__(self, inFeatures):
		super(ResidualBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
		)
	def forward(self, x):
		return x + self.block(x)

# GeneratorResNet class
class GeneratorResNet(nn.Module):
	def __init__(self, inputShape, numResidualBlocks):
		super(GeneratorResNet, self).__init__()
		channels = inputShape[0]

		outFeatures = 64
		model = [
			nn.ReflectionPad2d(3),
			nn.Conv2d(channels, outFeatures, 7),
			nn.InstanceNorm2d(outFeatures),
			nn.LeakyReLU(0.2, inplace=True),
		]
		for _ in range(2):
			inFeatures, outFeatures = outFeatures, outFeatures*2
			model += [
				nn.Conv2d(inFeatures, outFeatures, 3, 2, 1),
				nn.InstanceNorm2d(outFeatures),
				nn.ReLU(inplace=True),
			]

		for _ in range(numResidualBlocks):
			model += [ResidualBlock(outFeatures)]

		for _ in range(2):
			inFeatures, outFeatures = outFeatures, outFeatures//2
			model += [
				nn.Upsample(scale_factor=2),
				nn.Conv2d(inFeatures, outFeatures, 3, 1, 1),
				nn.InstanceNorm2d(outFeatures),
				nn.ReLU(inplace=True),
			]

		model += [
			nn.ReflectionPad2d(channels), 
			nn.Conv2d(outFeatures, channels, 7),
			nn.Tanh()
		]
		self.model = nn.Sequential(*model)
	
	def forward(self, x):
		return self.model(x)

# Discriminator class
class Discriminator(nn.Module):
	def __init__(self, inputShape):
		super(Discriminator, self).__init__()
		channels, height, width = inputShape

		self.outputShape = (1, height//2**4, width//2**4)

		self.model = nn.Sequential(
			*Discriminator.discriminatorBlock(channels, 64, False),
			*Discriminator.discriminatorBlock(64, 128),
			*Discriminator.discriminatorBlock(128, 256),
			*Discriminator.discriminatorBlock(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 1, 4, padding=1)
		)

	@staticmethod
	def discriminatorBlock(inFilters, outFilters, normalize=True):
		layers = [nn.Conv2d(inFilters, outFilters, 4, stride=2, padding=1)]
		if normalize: layers += [nn.InstanceNorm2d(outFilters)]
		layers += [nn.LeakyReLU(0.2, inplace=True)]
		return layers

	def forward(self, img):
		return self.model(img)

Channels = 3
ImgHeight, ImgWidth = 256, 256
ResidualBlocksAB = 7
ResidualBlocksBA = 13
LearningRate = 0.0002
BetaTuple = (0.5, 0.999)
Epochs = 500
InitEpoch = 0
DecayEpoch = 100
LambdaCyc = 10.0
LambdaID = 5.0
BatchSize = 4
SampleInterval = 10000
CheckPointInterval = 5

os.makedirs(f"images", exist_ok=True)
os.makedirs(f"saved_models", exist_ok=True)

criterionGAN = nn.MSELoss().to(device)
criterionCycle = nn.L1Loss().to(device)
criterionIdentity = nn.L1Loss().to(device)

inputShape = (Channels, ImgHeight, ImgWidth)
GAB = GeneratorResNet(inputShape, ResidualBlocksAB).to(device)
GBA = GeneratorResNet(inputShape, ResidualBlocksBA).to(device)
DA = Discriminator(inputShape).to(device)
DB = Discriminator(inputShape).to(device)

GAB.apply(initWeightsNormal)
GBA.apply(initWeightsNormal)
DA.apply(initWeightsNormal)
DB.apply(initWeightsNormal)

optimizerG = torch.optim.Adam(
	itertools.chain(GAB.parameters(), GBA.parameters()), lr=LearningRate,
	betas=BetaTuple
)
optimizerDA = torch.optim.Adam(
	DA.parameters(), lr=LearningRate, betas=BetaTuple)
optimizerDB = torch.optim.Adam(
	DB.parameters(), lr=LearningRate, betas=BetaTuple)

class LambdaLR:
	def __init__(self, epochs, offset, decayStartEpoch):
		self.denum = epochs - decayStartEpoch
		self.offset = offset
		self.decayStartEpoch = decayStartEpoch
	def step(self, epoch):
		return 1.0 - max(0, epoch+self.offset-self.decayStartEpoch)/self.denum

for d in os.listdir("saved_models/"):
	if not d.startswith("GAB") or not d.endswith(".pth"): continue
	epoch = int(d[3:-4])
	if epoch > InitEpoch: InitEpoch = epoch

if InitEpoch > 0:
	print(f"Load epoch {InitEpoch}....")
	path = "saved_models"
	GAB.load_state_dict(torch.load(f"{path}/GAB{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	GBA.load_state_dict(torch.load(f"{path}/GBA{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	DA.load_state_dict(torch.load(f"{path}/DA{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	DB.load_state_dict(torch.load(f"{path}/DB{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	optimizerG.load_state_dict(torch.load(f"{path}/optimizerG{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	optimizerDA.load_state_dict(torch.load(f"{path}/optimizerDA{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))
	optimizerDB.load_state_dict(torch.load(f"{path}/optimizerDB{InitEpoch:03}.pth", 
		map_location=device, weights_only=True, ))

lrSchedulerG = torch.optim.lr_scheduler.LambdaLR(
	optimizerG, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDA = torch.optim.lr_scheduler.LambdaLR(
	optimizerDA, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDB = torch.optim.lr_scheduler.LambdaLR(
	optimizerDB, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)

trainTransform = [
	transforms.Resize(int(ImgHeight*1.12), transforms.InterpolationMode.BICUBIC),
	transforms.RandomCrop((ImgHeight, ImgWidth)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
valTransform = [
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataLoader = DataLoader(
	ImageDataset(f"dataset", trainTransform, ),
	batch_size=BatchSize,
	shuffle=True,
)

valDataLoader = DataLoader(
	ImageDataset(f'dataset', valTransform, mode='test', ),
	batch_size=5,
	shuffle=True,
)

def sampleImages(batchesDone):
	GAB.eval()
	GBA.eval()
	with torch.no_grad():
		imgs = next(iter(valDataLoader))
		realA = imgs['A'].to(device)
		fakeB = GAB(realA)
		realB = imgs['B'].to(device)
		fakeA = GBA(realB)

		realA = make_grid(realA, nrow=5, normalize=True)
		realB = make_grid(realB, nrow=5, normalize=True)
		fakeA = make_grid(fakeA, nrow=5, normalize=True)
		fakeB = make_grid(fakeB, nrow=5, normalize=True)

		imageGrid = torch.cat((realA, fakeB, realB, fakeA), 1)
		save_image(imageGrid, f"images/{batchesDone:08}.png", normalize=False)
	GAB.train()
	GBA.train()

timeStamp, epochSize = time.time(), len(dataLoader)*BatchSize
epoch = InitEpoch
# make valid and fake target
valid = torch.ones((BatchSize, *DA.outputShape), device=device)
fake = torch.zeros((BatchSize, *DA.outputShape), device=device)
while epoch < Epochs:
	for i, batch in enumerate(dataLoader):
		# get images from data set
		realA = batch['A'].to(device)
		realB = batch['B'].to(device)

		optimizerG.zero_grad()

		lossIDA = criterionIdentity(GBA(realA), realA)
		lossIDB = criterionIdentity(GAB(realB), realB)
		lossIdentity = (lossIDA + lossIDB)/2

		fakeB = GAB(realA)
		fakeA = GBA(realB)
		lossGANAB = criterionGAN(DB(fakeB), valid)
		lossGANBA = criterionGAN(DA(fakeA), valid)
		lossGAN = (lossGANAB + lossGANBA)/2

		recovA = GBA(fakeB)
		recovB = GAB(fakeA)
		lossCycleA = criterionCycle(recovA, realA)
		lossCycleB = criterionCycle(recovB, realB)
		lossCycle = (lossCycleA + lossCycleB)/2

		lossG = lossGAN + LambdaCyc * lossCycle + LambdaID * lossIdentity

		lossG.backward()
		optimizerG.step()

		optimizerDA.zero_grad()
		lossReal = criterionGAN(DA(realA), valid)
		lossFake = criterionGAN(DA(fakeA.detach()), fake)
		lossDA = (lossReal + lossFake)/2
		lossDA.backward()
		optimizerDA.step()

		optimizerDB.zero_grad()
		lossReal = criterionGAN(DB(realB), valid)
		lossFake = criterionGAN(DB(fakeB.detach()), fake)
		lossDB = (lossReal + lossFake)/2
		lossDB.backward()
		optimizerDB.step()

		lossD = (lossDA + lossDB)/2

		batchesDone = epoch*epochSize + (i+1)*BatchSize
		batchesLeft = Epochs*epochSize - batchesDone
		deltaTime = (time.time()-timeStamp)/((epoch-InitEpoch)*epochSize+(i+1)*BatchSize)
		timeLeft = datetime.timedelta(seconds=batchesLeft*deltaTime)

		print(f"\r[{epoch+1}/{Epochs}][{i+1}/{len(dataLoader)}]",
			f"D:{lossD.item():.2f},",
			f"GAN:{lossG.item():.2f},",
			f"Cyc:{lossCycle.item():.2f},",
			f"Id:{lossIdentity.item():.2f}",
			f"ETA:{str(timeLeft)[:-7]}", flush=True)

		if batchesDone%SampleInterval == 0:
			sampleImages(batchesDone)

	lrSchedulerG.step()
	lrSchedulerDA.step()
	lrSchedulerDB.step()

	epoch += 1
	if epoch%CheckPointInterval == 0:
		path = "saved_models"
		torch.save(GAB.state_dict(), f"{path}/GAB{epoch:03}.pth")
		torch.save(GBA.state_dict(), f"{path}/GBA{epoch:03}.pth")
		torch.save(DA.state_dict(), f"{path}/DA{epoch:03}.pth")
		torch.save(DB.state_dict(), f"{path}/DB{epoch:03}.pth")
		torch.save(optimizerG.state_dict(), f"{path}/optimizerG{epoch:03}.pth")
		torch.save(optimizerDA.state_dict(), f"{path}/optimizerDA{epoch:03}.pth")
		torch.save(optimizerDB.state_dict(), f"{path}/optimizerDB{epoch:03}.pth")
