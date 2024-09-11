import torch # PyTorch kütüphanesini içe aktar
import torch.nn as nn # PyTorch'un modülü olan nn'i içe aktar
import torchvision.datasets as datasets # PyTorch'un veri setlerini içe aktar
import torchvision.transforms as transforms # PyTorch'un dönüşümlerini içe aktar
from torch.autograd import Variable # Değişkenleri içe aktar
from torch.utils.data import DataLoader, TensorDataset # DataLoader ve TensorDataset'i içe aktar


import matplotlib.pyplot as plt # Grafik çizdirmek için matplotlib'ı içe aktar
from PIL import Image # Resimleri açmak için PIL kütüphanesini içe aktar
import numpy as np # Dizilerle işlem yapmak için numpy kütüphanesini içe aktar


#precomputed mean and std values for MNIST dataset
# MNIST veri setinin ortalama değeri
mean_grat = 0.1307 
std_grat = 0.3081 

# Dönüşümleri tanımla
transforms_orginals = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean_grat,), (std_grat,))])


transforms_photo = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((mean_grat,), (std_grat,))])


# Eğitim veri setini yükle
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_orginals) 

# Test veri setini yükle
test_dataset = datasets.MNIST(root='./data', train=False,transform=transforms_orginals)

# NumPy dizisi ve dönüşümleri tanımla
random_image = train_dataset[20][0].numpy() * std_grat + mean_grat

# Resmi çizdir
plt.imshow(random_image.reshape(28,28), cmap='gray')
plt.title('MNIST Image')
plt.savefig('mnist_image.png')
plt.close()
