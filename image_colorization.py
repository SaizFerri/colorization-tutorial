#!/usr/bin/env python3
import os
import sys
import argparse
import time
import copy
import shutil

# from PIL import Image
#from IPython.display import Image, display

import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
  dest='data_dir', type=str,
  help='Data: Path to read-only directory containing image *.jpeg files.'
)

parser.add_argument(
  '--outputs-dir', type=str, default=None,
  help='Data: Path to writable directory for the output images'
)

parser.add_argument(
  '--checkpoints-dir', type=str, default=None,
  help='Data: Path to writable directory for the checkpoint files'
)

parser.add_argument(
  '--learning-rate', type=float, default=0.001,
  help='Training: Learning rate. Default: 0.001'
)

parser.add_argument(
  '--batch-size', type=int, default=64,
  help='Training: Batch size. Default: 64'
)

parser.add_argument(
  '--num-epochs', type=int, default=100,
  help='Training: Number of epochs. Default: 100'
)

parser.add_argument(
  '--log-dir', type=str, default=None,
  help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)

parser.add_argument(
  '--log-file-name', type=str, default='training.log',
  help='Debug: Name of the log file, generated when --log-dir is set. Default: training.log'
)

args = parser.parse_args()

# Redirect output streams for logging
if args.log_dir:
  log_file = open(os.path.join(os.path.expanduser(args.log_dir), args.log_file_name), 'w')
  sys.stdout = log_file
  sys.stderr = log_file

data_dir = os.path.expanduser(args.data_dir)

TRAIN_PATH = os.path.join(data_dir, 'images/train')
VAL_PATH = os.path.join(data_dir, 'images/val')

OUTPUT_GRAY_PATH = os.path.join(os.path.expanduser(args.outputs_dir), 'gray/')
OUTPUT_COLOR_PATH = os.path.join(os.path.expanduser(args.outputs_dir), 'color/')

CHECKPOINTS_PATH = os.path.expanduser(args.checkpoints_dir)

'''
  Needed to create the folders with the data after unziping
'''
# os.makedirs('images/train/class/', exist_ok=True) # 40,000 images
# os.makedirs('images/val/class/', exist_ok=True)   #  1,000 images

# for i, file in enumerate(os.listdir('testSet_resize')):
#   if i < 1000: # first 1000 will be val
#     os.rename('testSet_resize/' + file, 'images/val/class/' + file)
#   else: # others will be val
#     os.rename('testSet_resize/' + file, 'images/train/class/' + file)

'''
---------------------------------------------------------------------------
'''

# Image.open('images/val/class/ebfca8b4255f54756d62fe0a50e78733.jpg').show()

use_gpu = torch.cuda.is_available()

class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(num_classes=365)
    # print(resnet)
    # Change first conv layer to accept sinle-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output

model = ColorizationNet()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)


class GrayscaleImageFolder(datasets.ImageFolder):
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()

    if self.target_transform is not None:
      target = self.target_transform(target)
    
    return img_original, img_ab, target

# Training
train_transforms = transforms.Compose([
  transforms.RandomResizedCrop(224),
  transforms.RandomHorizontalFlip()
])
train_imagefolder = GrayscaleImageFolder(TRAIN_PATH, train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True)

# Validation
val_transforms = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224)
])
val_imagefolder = GrayscaleImageFolder(VAL_PATH, val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=args.batch_size, shuffle=False)

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''
    Show/save rgb image from grayscale and ab channels
    Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}
  '''
  plt.clf()
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() #combine channels
  color_image = color_image.transpose((1, 2, 0)) # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()

  if save_path is not None and save_name is not None:
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'grayscale': OUTPUT_GRAY_PATH, 'colorized': OUTPUT_COLOR_PATH}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg

def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training epoch {}'.format(epoch))
  model.train()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (input_gray, input_ab, target) in enumerate(train_loader):
    
    # Use GPU if available
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Record time to load data (above)
    data_time.update(time.time() - end)

    # Run forward pass
    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item()*100, input_gray.size(0))

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record time to do forward and backward passes
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to value, not validation
    if i % 25 == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses)) 

  print('Finished training epoch {}'.format(epoch))

#checkpoint = torch.load('checkpoints/model-epoch-1-losses-0.003.pth')
#model.load_state_dict(checkpoint)

# Move model and loss function to GPU
if use_gpu: 
  criterion = criterion.cuda()
  model = model.cuda()

# Make folders and set parameters
# os.makedirs('outputs/color', exist_ok=True)
# os.makedirs('outputs/gray', exist_ok=True)
# os.makedirs('checkpoints', exist_ok=True)
save_images = True
best_losses = 2
epochs = args.num_epochs

for epoch in range(0, epochs):
  # Train for one epoch, then validate
  train(train_loader, model, criterion, optimizer, epoch)
  with torch.no_grad():
    losses = validate(val_loader, model, criterion, save_images, epoch)
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    torch.save(model.state_dict(), '{}/model-{}-{:.3f}.pth'.format(CHECKPOINTS_PATH, epoch+1,losses))

# validate(val_loader, model, criterion, True)

if args.log_dir:
  sys.stdout.close()