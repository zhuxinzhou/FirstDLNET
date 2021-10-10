# 使Pytorch 训练一个深度学习网络 #
> 朱心洲  22151326 
*   网络结构：卷积神经网络（CNN）
*   使用数据集：FER2013
*   运行方式:
*   正确率：60%+
*   所选框架：Pytorch
*   模型训练平台：Google Colab
*   完成日期： 2021.10.10
## 1. 作业介绍：
- Creating your own github account.
- Implementing your own deep neural network (in Pytorch, PaddlePaddle...).
- Training it on CIFAR10.
- Tuning a hyper-parameter and analyzing its effects on performance.Writing a README.md to report your findings
## 2. 实现过程：
### 2.1 数据集的导入
```
from torchvision import datasets,transforms
'''
CIFAR-10是kaggle计算机视觉竞赛的一个图像分类项目。
该数据集共有60000张32*32彩色图像，一共可以分为"plane", “car”, “bird”,“cat”, “deer”, “dog”, “frog”,“horse”,“ship”, “truck” 10类，
每类6000张图。有50000张用于训练，构成了5个训练批，每一批10000张图；10000张用于测试，单独构成一批。
tarnsform_fun=transforms.Compose([transforms.Pad(4),transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
#一般用Compose把多个步骤整合到一起
#1.缩放图片，变成32*32
#2.归一化至0-1
#3.归一化到-1，1``input[channel] = (input[channel] - mean[channel]) / std[channel]``
training_data=datasets.CIFAR10(root='./data',train=True,download=True,transform=tarnsform_fun)

validation_data=datasets.CIFAR10(root='./data',train=False,download=True,transform=tarnsform_fun)
train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
```
CIFAR-10为3*32*32的彩色图片，如下图所示
![部分图片](https://upload-images.jianshu.io/upload_images/16487280-18567af2d75b7c45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 2.2 定义一些超参数
```
#定义hyper-parameter
num_epochs = 10
num_classes = 10
batch_size = 32
learning_rate = 0.1
```
### 2.3 构建网络
数据集数据较复杂，特征较难提取，使用卷积神经网络实现，
 ```
两个卷积层
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积核5*5
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(16),
            #ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(8*8*32, num_classes)
        
    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

### 2.4 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
```
交叉熵主要是用来判定实际的输出与期望的输出的接近程度，为什么这么说呢，
举个例子：在做分类的训练的时候，如果一个样本属于第K类，那么这个类别所对应的的输出节点的输出值应该为1，
而其他节点的输出都为0，即[0,0,1,0,….0,0]，这个数组也就是样本的Label，
是神经网络最期望的输出结果。也就是说用它来衡量网络的输出与标签的差异，利用这种差异经过反向传播去更新网络参数。

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
### 2.5 模型的训练
读入数据集，对模型进行训练
```
for i, (images, labels) in enumerate(train_loader):
        # 注意模型在GPU中，数据也要搬到GPU中
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
### 2.6 相应结果可视化
将训练损失，校验集损失，训练准确率，校验准确率通过折线图的方式绘出
![实验结果(调参前)](https://upload-images.jianshu.io/upload_images/16487280-ae186586dacbd85f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
_通过计算，在10轮训练结束后，训练准确率在39%，测试集准确率在43%_

##3. 超参数的调整
### 3.1 学习率的调整
通过分析，训练过程中有过拟合的迹象，学习率可能过高，因此，调整学习率至0.001，调整学习率后，重新训练，评估模型
```
learning_rate = 0.001
```
在调整学习率后，训练结果有明显优化，如下图所示。
![训练结果（调整学习率后）](https://upload-images.jianshu.io/upload_images/16487280-4fcc5467517f2342.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
_在5轮训练后，训练准确率在68%，测试集准确率在66%，有明显提高_
### 3.2 网络层数的调整
为了更好的捕捉图形的特征，通过增加卷积网络层数。由两层增加至三层
```angular2html
# 3个卷积层

class CNNpro(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNpro, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(16),
            #ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        self.fc = nn.Linear(4*4*64, num_classes)
        
    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```
最终结果：
![训练结果（调整网络层数后）](https://upload-images.jianshu.io/upload_images/16487280-24e5072c2e14f892.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
_在5轮训练后，训练准确率在90%，测试集准确率在72%，有明显提高_
## 总结
>在本次作业中，我第一次用pytorch实现了一个神经网络模型，并针对cifar10进行了训练,在初次训练，模型准确率在40%左右，后面通过针对学习率、网络层的调整，最终准确度达到了75%
