import os
from PIL import Image
import numpy as np
import random

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class TNBCDataset(Dataset):
	def __init__(self, root):
		self.root = root
		self.img_list = os.listdir(os.path.join(root, 'images'))
		#self.transform = transform

	def __len__(self):
		return len(self.img_list)
	
	def transform(self, image, mask):
        # 
		'''
		resize = transforms.Resize(size=(520, 520))
		image = resize(image)
		mask = resize(mask)
		'''

		# Random crop
		i, j, h, w = transforms.RandomCrop.get_params(
			image, output_size=(512, 512))
		image = TF.crop(image, i, j, h, w)
		mask = TF.crop(mask, i, j, h, w)

		# Random horizontal flipping
		if random.random() > 0.5:
			image = TF.hflip(image)
			mask = TF.hflip(mask)

		# Random vertical flipping
		if random.random() > 0.5:
			image = TF.vflip(image)
			mask = TF.vflip(mask)

		# Transform to tensor
		image = TF.to_tensor(image)
		mask = TF.to_tensor(mask)
		return image, mask
	

	def __getitem__(self, idx):

		img = Image.open(os.path.join(self.root, 'images', self.img_list[idx])).convert('RGB')
		mask = Image.open(os.path.join(self.root, 'labels', self.img_list[idx])).convert('L')

		img, mask = self.transform(img, mask)
		return img, mask

