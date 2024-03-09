import torch
import torch.nn as nn
from torchvision import datasets                                                                                         #for reading from the dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets                                                                                #torchvision has the CIFAR10 dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import  torch.optim as optim                                                                                                # for optimizer
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch, torchvision
from torchvision import models


import sys
print(sys.version)



train_ds=datasets.CIFAR10("data",train= True,download=True,transform = transforms.ToTensor())
test_ds=datasets.CIFAR10("data",train= False,download=True,transform = transforms.ToTensor())

train_ds.data.shape

test_ds.data.shape

batch_size=32                                                                                                                                               ##  should divide data to some parts for model using Dataloader , then use the parts in train and test
train_dl=DataLoader(dataset=train_ds,batch_size=batch_size,shuffle=True,num_workers=2)                 #shuffle : for select random pics / num_workers: ready next pics in the same : increase the speed for loading
test_dl=DataLoader(dataset=test_ds,batch_size=batch_size,shuffle=True,num_workers=2)



image,target=next (iter(train_dl) )
plt.figure(figsize=(8,8))
for i in range(18):
     plt.subplot(3,6,i+1)
     img=torch.transpose(image[i],0,1)                                                                                 #   کانالهای رنگی را باید تغییر دهیم یعنی جای بیتها تغییر کند
     img=torch.transpose(img,1,2)                                                                                        #تغییر بیتهای آر و جی و بی / جای بیت 0و1و2 تغییر میکند
     plt.imshow(img)
     plt.axis( 'off' )
plt.show()


# define CNN algorithm
class CNN(nn.Module):
  def __init__(self,input_channel=3,num_class=10):                                                                                                        # 3 is R,G,B
    super(CNN,self).__init__()
    self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=16,kernel_size=(3,3),padding=(1,1))                  #define first layer  16تا فیلتر داره اعمال میشه
    self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=(1,1))                                     #define second layer / kernel_size: number f filter / padding : برای اینکه تصویر در هر مرحله کوچک نشه
    self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=(1,1))                                     #define third layer
    self.pool=nn.MaxPool2d(kernel_size=(2,2))  #پولینک را اعمال میکنیم با هربار اعمال لایه پولینگ طول تصویر نصف میشود
    self.fc1=nn.Linear(in_features=16*16*64,out_features=num_class)                                                                           # linear layer for output

  def forward(self,x):  #define out layers
    out=self.conv1(x)     # خروجی لایه اول اینجا قرار میگیرد
    out=self.conv2(out)
    out=self.pool(out)
    out=self.conv3(out)
    out=out.reshape(out.shape[0],-1)    #تبدیل به حالت برداری
    out=self.fc1(out)    # به لایه فولی کانکتت میدهیم تا یکسری امتیازات بهمون بدهد
    return out


device='cuda' if torch.cuda.is_available() else 'cpu'   #using of cpu or gpu
device

#define CNN algorithm
model=CNN().to(device)
model

print(model)



model_SGD=model
#model_RMSprop=model
#model_Adam=model


# loss
citerion=nn.CrossEntropyLoss()


#optimizer
optimizer_SGD=torch.optim.SGD(params=model_SGD.parameters(),lr=0.01)
#optimizer_RMSprop=torch.optim.RMSprop(params=model_RMSprop.parameters(),lr=0.01)
#optimizer_Adam=torch.optim.Adam(params=model_Adam.parameters(),lr=0.01)


epoch=5


#train using SGD
for i in range(epoch):
      sumLoss=0
      for idx,(image,target) in enumerate(train_dl,0):    # در هر مرحله تصاویر را برمیگردونه به ایمیج و تارگت

            image=image.to(device)
            target=target.to(device)

            optimizer_SGD.zero_grad()    #optimizering in every gradiant گرادیانها را صفر میکنیم

            score_SGD=model_SGD(image)     #getting score from the training models
            loss=citerion(score_SGD,target)       # مقایسه بین تصویر و تارگت انجام میدهد و در این قسمت قرار میدهد

            sumLoss+=loss                                  #تعداد تشابه ها

            loss.backward()                                # calculate the gradiant

            optimizer_SGD.step()                     # update the calculating parameters in models


      print(f' in epoch number {i+1} is equal to { sumLoss }'  )

#check_evaluate_SGD for accuracy,precision,recall,F1-score : evaluation criteria
def check_evaluate_SGD(dataloader,model_SGD):
      if dataloader.dataset.train:
           print('accuracy SGD on train data is calculating...')
      else:
           print('accuracy SGD on test data is calculating...')

      true_positive=0
      false_positive=0
      total=0
      model_SGD.eval()
      with torch.no_grad():
             for x_SGD,y_SGD in dataloader:    #برای ارزیابی مدل
                   x_SGD=x_SGD.to(device)
                   y_SGD=y_SGD.to(device)

                   score_SGD=model_SGD(x_SGD)
                   _,pred_SGD=score_SGD.max(1)    # get the score in every dimention 1
                   true_positive+=(pred_SGD==y_SGD).sum()   #مجموع آنهایی که با تارگت درست تشخیص داده شدند
                   false_positive+=(pred_SGD!=y_SGD).sum()
                   total+=len(y_SGD)
                   ######
                   accuracy=true_positive/total
                   precision=true_positive/(true_positive+false_positive)
                   #Recall=true_positive/(true_positive+false_negative)
                   #F1_score= ( 2*(precision*Recall)+(precision+Recall) )

      print(f'acuracy SGD is { accuracy }')
      print(f'precision SGD is { precision  }')
      #print(f' Recall SGD is {Recall}')
      #print(f' F1-score SGD is {2*(precision*Recall)+(precision+Recall)}')

check_evaluate_SGD(test_dl,model_SGD)

check_evaluate_SGD(train_dl,model_SGD)