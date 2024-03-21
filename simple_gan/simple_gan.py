#GAN (Generative adversial networks)
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio #initial with "pip install imageio"
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

#create folders
if not os.path.exists('./checkpoint'):
  os.mkdir('./checkpoint')
if not os.path.exists('./dataset'):
  os.mkdir('./dataset')
if not os.path.exists('./img'):
  os.mkdir('./img')
if not os.path.exists('./img/real'):
  os.mkdir('./img/real')
if not os.path.exists('./img/fake'):
  os.mkdir('./img/fake')
#define visualization & image saving code

#visualiza the first image from the torch tensor
def vis_image(image):
  plt.imshow(image[0].detach().cpu.numpy(),cmap='grey')
  plt.show()
def save_gif(training_progress_images,images):
  img_grid=make_grid(images.data)
  img_grid=np.transpose(img_grid.detach().cpu().numpy(),(1,2,0))
  img_grid=255. * img_grid
  img_grid=img_grid.astype(np.uint8)
  training_progress_images.append(img_grid)
  imageio.mimsave('./img/training_progress.gif',training_progress_images)
  return training_progress_images

def vis_gif(training_progress_images): #visualize gif file
  fig=plt.figure()
  ims=[]
  for i in range(len(training_progress_images)):
    im=plt.imshow(training_progress_images[i],animated=True)
    ims.append([im])
  ani= animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
  html=ani.to_html5_video()
  HTML(html)
def plot_gif(training_progress_images,plot_length=10):
  plt.close()
  fig=plt.figure()
  total_len=len(training_progress_images)
  for i in range(plot_length):
    im=plt.imshow(training_progress_images[int(total_len/plot_length)*i])
    plt.show()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_imagelist(dataset, real):
    if real:
        base_path = './img/real'
    else:
        base_path = './img/fake'
    # Ensure the directory exists
    ensure_dir(base_path)
    dataset_paths = []
    for i, image in enumerate(dataset):
        save_path = f'{base_path}/image_{i}.png'
        dataset_paths.append(save_path)
        vutils.save_image(image.cpu(), save_path)
    return dataset_paths
#load dataset,define dataloader
dataset= dset.MNIST(root="./dataset", download=True,transform=transforms.Compose([transforms.ToTensor(),]))
dataloader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True,num_workers=2)
#define your generator & discriminator
#define generator module
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.main=nn.Sequential(nn.Linear(100,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,784),nn.Sigmoid(),)

  def forward(self,input):
    output=self.main(input)
    output=output.view(-1,1,28,28)
    return output

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.main=nn.Sequential(nn.Linear(28*28,256),nn.LeakyReLU(0.2),nn.Linear(256,256),nn.LeakyReLU(0.2),nn.Linear(256,1),nn.Sigmoid(),)

  def forward(self,input):
    input=input.view(-1,28*28)
    output=self.main(input)
    output=output.squeeze(dim=1)
    return output

#upload on GPU,define optimizer
netG= Generator() 
netD=Discriminator() 
optimizerD= optim.Adam(netD.parameters(),lr=0.0002)
optimizerG=optim.Adam(netG.parameters(),lr=0.0002)
#noise sampling
noise=torch.randn(128,100)
#####################################################
fixed_noise=torch.randn(128,100) 
criterion=nn.BCELoss()
n_epoch=3
training_progress_images_list=[]
for epoch in range(n_epoch):
  print(f'Epoch [{epoch + 1}/{n_epoch}] Started.')
  for i, (data,_) in enumerate(dataloader):
    #update D network: maximize log(D(x))+log(1-D(G(z)))
    #train with real
    netD.zero_grad()
    data= data 
    batch_size=data.size(0)
    label=torch.ones((batch_size,))  #real label=1
    output=netD(data)
    errD_real=criterion(output,label)
    D_x = output.mean().item()

    #train with fake
    noise=torch.randn(batch_size,100) 
    fake=netG(noise)
    label=torch.zeros((batch_size,))  #fake label=0
    output=netD(fake.detach())
    errD_fake=criterion(output,label)
    D_G_zl=output.mean().item()

    #loss backward
    errD=errD_real + errD_fake
    errD.backward()
    optimizerD.step()

    ##############################################################
    #now update G network: maximize log(D(G(z)))
    netG.zero_grad()
    label=torch.ones((batch_size,))  #fake labels are real for generator cost
    output=netD(fake)
    errG=criterion(output,label)
    D_G_z2= output.mean().item()



import pytorch_fid
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid import inception

#evaluate your model (save samples)
test_dataset=dset.MNIST(root="./dataset", download=True,train=False,transform=transforms.Compose([transforms.ToTensor(),]))
dataloader= torch.utils.data.DataLoader(test_dataset,batch_size=50,shuffle=True,num_workers=2)

for i,(data,_) in enumerate(dataloader):
  real_dataset=data
  break

noise= torch.randn(50,100)
fake_dataset=netG(noise)

real_image_path_list= save_imagelist(real_dataset,True)
false_image_path_list= save_imagelist(fake_dataset,False)


real_images_dir = './img/real'
fake_images_dir = './img/fake'

#Evaluate FID score
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fid_value = calculate_fid_given_paths([real_images_dir, fake_images_dir], 28, device, 2048)
print(f'FID scorep: {fid_value}')