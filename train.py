"""
Title: Cycle GAN
Description: Cycle GAN
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

# Enable hardware acceleration
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def load_image(file_name):
    """Load an image and ensure it's in RGB format."""
    image = Image.open(file_name)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

# Dataset class for loading images
class ImageDataset(Dataset):
    def __init__(self, root, dir_a, dir_b, trans, unaligned=True, mode='train'):
        self.transform = transforms.Compose(trans)
        self.unaligned = unaligned
        self.files_a = sorted(glob.glob(os.path.join(root, f'{dir_a}/{mode}') + '/*.*'))
        self.files_b = sorted(glob.glob(os.path.join(root, f'{dir_b}/{mode}') + '/*.*'))
        self.len_a = len(self.files_a)
        self.len_b = len(self.files_b)

    def __getitem__(self, index):
        idx_a = index % self.len_a
        image_a = load_image(self.files_a[idx_a])
        idx_b = random.randrange(self.len_b) if self.unaligned else index % self.len_b
        image_b = load_image(self.files_b[idx_b])
        item_a = self.transform(image_a)
        item_b = self.transform(image_b)
        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return self.len_a

# Initialize weights using a normal distribution
def init_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# Residual Block for the Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

# Generator with ResNet architecture
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Downsampling
        for _ in range(2):
            in_features, out_features = out_features, out_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, 2, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        # Upsampling
        for _ in range(2):
            in_features, out_features = out_features, out_features // 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, 1, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
        # Final output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        self.output_shape = (1, height // 2**4, width // 2**4)

        self.model = nn.Sequential(
            *Discriminator.discriminator_block(channels, 64, False),
            *Discriminator.discriminator_block(64, 128),
            *Discriminator.discriminator_block(128, 256),
            *Discriminator.discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    @staticmethod
    def discriminator_block(in_filters, out_filters, normalize=True):
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalize:
            layers += [nn.InstanceNorm2d(out_filters)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        return layers

    def forward(self, img):
        return self.model(img)

# Hyperparameters and configurations
channels = 3
img_height, img_width = 256, 256
residual_blocks_ab = 7
residual_blocks_ba = 13
learning_rate = 0.0002
beta_tuple = (0.5, 0.999)
epochs = 500
init_epoch = 0
decay_epoch = 100
lambda_cyc = 10.0
lambda_id = 5.0
batch_size = 4
sample_interval = 10000
checkpoint_interval = 5

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Define loss functions
criterion_gan = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)

# Initialize models
input_shape = (channels, img_height, img_width)
generator_ab = GeneratorResNet(input_shape, residual_blocks_ab).to(device)
generator_ba = GeneratorResNet(input_shape, residual_blocks_ba).to(device)
discriminator_a = Discriminator(input_shape).to(device)
discriminator_b = Discriminator(input_shape).to(device)

# Apply weight initialization
generator_ab.apply(init_weights_normal)
generator_ba.apply(init_weights_normal)
discriminator_a.apply(init_weights_normal)
discriminator_b.apply(init_weights_normal)

# Optimizers
optimizer_g = torch.optim.Adam(
    itertools.chain(generator_ab.parameters(), generator_ba.parameters()),
    lr=learning_rate,
    betas=beta_tuple
)
optimizer_da = torch.optim.Adam(
    discriminator_a.parameters(), lr=learning_rate, betas=beta_tuple
)
optimizer_db = torch.optim.Adam(
    discriminator_b.parameters(), lr=learning_rate, betas=beta_tuple
)

# Learning rate schedule
class LambdaLR:
    def __init__(self, epochs, offset, decay_start_epoch):
        self.denum = epochs - decay_start_epoch
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / self.denum

# Check for saved models and load the latest epoch if available
for file in os.listdir("saved_models/"):
    if file.startswith("GAB") and file.endswith(".pth"):
        epoch = int(file[3:-4])
        if epoch > init_epoch:
            init_epoch = epoch

if init_epoch > 0:
    print(f"Loading models from epoch {init_epoch}...")
    path = "saved_models"
    generator_ab.load_state_dict(torch.load(f"{path}/GAB{init_epoch:03}.pth", map_location=device))
    generator_ba.load_state_dict(torch.load(f"{path}/GBA{init_epoch:03}.pth", map_location=device))
    discriminator_a.load_state_dict(torch.load(f"{path}/DA{init_epoch:03}.pth", map_location=device))
    discriminator_b.load_state_dict(torch.load(f"{path}/DB{init_epoch:03}.pth", map_location=device))
    optimizer_g.load_state_dict(torch.load(f"{path}/optimizerG{init_epoch:03}.pth", map_location=device))
    optimizer_da.load_state_dict(torch.load(f"{path}/optimizerDA{init_epoch:03}.pth", map_location=device))
    optimizer_db.load_state_dict(torch.load(f"{path}/optimizerDB{init_epoch:03}.pth", map_location=device))

lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
    optimizer_g, lr_lambda=LambdaLR(epochs, init_epoch, decay_epoch).step
)
lr_scheduler_da = torch.optim.lr_scheduler.LambdaLR(
    optimizer_da, lr_lambda=LambdaLR(epochs, init_epoch, decay_epoch).step
)
lr_scheduler_db = torch.optim.lr_scheduler.LambdaLR(
    optimizer_db, lr_lambda=LambdaLR(epochs, init_epoch, decay_epoch).step
)

# Data transformations
train_transform = [
    transforms.Resize(int(img_height * 1.12), transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
val_transform = [
    transforms.Resize((img_height, img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Load datasets
data_loader = DataLoader(
    ImageDataset("dataset", "face", "animeface", train_transform),
    batch_size=batch_size,
    shuffle=True,
)

val_data_loader = DataLoader(
    ImageDataset("dataset", "face", "animeface", val_transform, mode='test'),
    batch_size=5,
    shuffle=True,
)

# Sampling function to save images
def sample_images(batches_done):
    generator_ab.eval()
    generator_ba.eval()
    with torch.no_grad():
        imgs = next(iter(val_data_loader))
        real_a = imgs['A'].to(device)
        fake_b = generator_ab(real_a)
        real_b = imgs['B'].to(device)
        fake_a = generator_ba(real_b)

        real_a = make_grid(real_a, nrow=5, normalize=True)
        real_b = make_grid(real_b, nrow=5, normalize=True)
        fake_a = make_grid(fake_a, nrow=5, normalize=True)
        fake_b = make_grid(fake_b, nrow=5, normalize=True)

        image_grid = torch.cat((real_a, fake_b, real_b, fake_a), 1)
        save_image(image_grid, f"images/{batches_done:08}.png", normalize=False)
    generator_ab.train()
    generator_ba.train()

# Training loop
timestamp, epoch_size = time.time(), len(data_loader) * batch_size
epoch = init_epoch
valid = torch.ones((batch_size, *discriminator_a.output_shape), device=device)
fake = torch.zeros((batch_size, *discriminator_a.output_shape), device=device)

while epoch < epochs:
    for i, batch in enumerate(data_loader):
        real_a = batch['A'].to(device)
        real_b = batch['B'].to(device)

        # Train Generators
        optimizer_g.zero_grad()

        loss_id_a = criterion_identity(generator_ba(real_a), real_a)
        loss_id_b = criterion_identity(generator_ab(real_b), real_b)
        loss_identity = (loss_id_a + loss_id_b) / 2

        fake_b = generator_ab(real_a)
        fake_a = generator_ba(real_b)
        loss_gan_ab = criterion_gan(discriminator_b(fake_b), valid)
        loss_gan_ba = criterion_gan(discriminator_a(fake_a), valid)
        loss_gan = (loss_gan_ab + loss_gan_ba) / 2

        recov_a = generator_ba(fake_b)
        recov_b = generator_ab(fake_a)
        loss_cycle_a = criterion_cycle(recov_a, real_a)
        loss_cycle_b = criterion_cycle(recov_b, real_b)
        loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

        loss_g = loss_gan + lambda_cyc * loss_cycle + lambda_id * loss_identity
        loss_g.backward()
        optimizer_g.step()

        # Train Discriminator A
        optimizer_da.zero_grad()
        loss_real_a = criterion_gan(discriminator_a(real_a), valid)
        loss_fake_a = criterion_gan(discriminator_a(fake_a.detach()), fake)
        loss_da = (loss_real_a + loss_fake_a) / 2
        loss_da.backward()
        optimizer_da.step()

        # Train Discriminator B
        optimizer_db.zero_grad()
        loss_real_b = criterion_gan(discriminator_b(real_b), valid)
        loss_fake_b = criterion_gan(discriminator_b(fake_b.detach()), fake)
        loss_db = (loss_real_b + loss_fake_b) / 2
        loss_db.backward()
        optimizer_db.step()

        loss_d = (loss_da + loss_db) / 2

        batches_done = epoch * epoch_size + (i + 1) * batch_size
        batches_left = epochs * epoch_size - batches_done
        elapsed_time = (time.time() - timestamp) / ((epoch - init_epoch) * epoch_size + (i + 1) * batch_size)
        eta = datetime.timedelta(seconds=batches_left * elapsed_time)

        print(f"\r[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(data_loader)}]",
              f"[D loss: {loss_d:.4f}] [G loss: {loss_g:.4f}]",
              f"[Cycle loss: {loss_cycle:.4f}] [Identity loss: {loss_identity:.4f}] ETA: {eta}", flush=True)

        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    lr_scheduler_g.step()
    lr_scheduler_da.step()
    lr_scheduler_db.step()

    epoch += 1
    if epoch % checkpoint_interval == 0:
        path = "saved_models"
        torch.save(generator_ab.state_dict(), f"{path}/GAB{epoch:03}.pth")
        torch.save(generator_ba.state_dict(), f"{path}/GBA{epoch:03}.pth")
        torch.save(discriminator_a.state_dict(), f"{path}/DA{epoch:03}.pth")
        torch.save(discriminator_b.state_dict(), f"{path}/DB{epoch:03}.pth")
        torch.save(optimizer_g.state_dict(), f"{path}/optimizerG{epoch:03}.pth")
        torch.save(optimizer_da.state_dict(), f"{path}/optimizerDA{epoch:03}.pth")
        torch.save(optimizer_db.state_dict(), f"{path}/optimizerDB{epoch:03}.pth")
