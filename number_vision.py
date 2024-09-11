import torch # PyTorch kütüphanesini içe aktar
import torch.nn as nn # PyTorch'un modülü olan nn'i içe aktar
import torchvision.datasets as datasets # PyTorch'un veri setlerini içe aktar
import torchvision.transforms as transforms # PyTorch'un dönüşümlerini içe aktar
from torch.autograd import Variable # Değişkenleri içe aktar
from torch.utils.data import DataLoader, TensorDataset # DataLoader ve TensorDataset'i içe aktar


import matplotlib.pyplot as plt # Grafik çizdirmek için matplotlib'ı içe aktar
from PIL import Image # Resimleri açmak için PIL kütüphanesini içe aktar
import numpy as np # Dizilerle işlem yapmak için numpy kütüphanesini içe aktar