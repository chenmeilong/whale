#Epoch 50. Train Loss: 0.003284, Train Acc: 1.000000, Valid Loss: 0.939177, Valid Acc: 0.891429, Time 00:10:28

import os
import time
import pandas as pd
import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from PIL import Image
from datetime import datetime
import use_fcn
import cv2
from sklearn import metrics
import numpy as np
plt_train_loss = []
plt_test_loss = []
plt_train_acc = []
plt_test_acc = []
#数据增强
data_transforms = {
    'train':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300,300), interpolation=2),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机左右翻转
        #transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomGrayscale(p=0.2),  # 随机变为灰度图像
        transforms.ColorJitter(brightness=0.3),  # ，1 表示原图  随机亮度
        transforms.ColorJitter(contrast=0.3),  # 随机从 0 ~ 2 之间对比度变化，1 表示原图
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化对神经网络
    ]),
    'test':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

def load_test_data(data_path):    #加载全部的测试数据集 ，这样可以提高运行效率  ，因为训练数据集需要数据增强，所以不方便这样加载
    csv_test_data = pd.read_csv(data_path+ "/test.csv", sep=',')  # 读取测试数据标签
    image_test_list = csv_test_data.iloc[:, 0].values
    label_test_list=csv_test_data.iloc[:, 1].values
    for i in range(csv_test_data.iloc[:, 0].size):
        im =cv2.imread(data_path+"/"+"test"+"/"+image_test_list[i])  # 打开图片
        imge = data_transforms['test'](im)
        imge=use_fcn.pre_fcn(imge)    # ###################加入fcn            mask
        imge = torch.unsqueeze(imge, 0)
        if i == 0:
            test_tensor = imge
        else:
            test_tensor = torch.cat([test_tensor, imge], 0)
    label_test_tensor = torch.Tensor(label_test_list)
    label_test_tensor = label_test_tensor.long()
    print("成功加载测试数据集：",test_tensor.shape)
    return test_tensor,label_test_tensor   #训练数据集标签，测试数据集标签      #全部存为tensor

def load_image(i, train_test, batch_size,data_dir):    #训练数据集加载 边训边加载
    ##读取csv的label和图片循环  打开图片resize、转换成tensor
    # 返回tensore 数据 和tensor 标签先用数字表示
    if train_test=="train":
        csv_data = pd.read_csv(data_dir + "/train.csv", sep=',')  # 读取训练数据标签
    else:
        csv_data = pd.read_csv(data_dir + "/test.csv", sep=',')  # 读取测试数据标签
    image_list = csv_data.iloc[:, 0]
    label_list=csv_data.iloc[:, 1]
    for j in range(batch_size):
        im = cv2.imread(data_dir+"/"+train_test+"/"+image_list.values[batch_size * i + j])  # 打开图片
        imge = data_transforms['train'](im)   #使用数据增强
        imge=use_fcn.pre_fcn(imge)    # ###################加入fcn            mask
        imge = torch.unsqueeze(imge, 0)
        if j == 0:
            data_tensor = imge
        else:
            data_tensor = torch.cat([data_tensor, imge], 0)
    #print(data_tensor.shape)
    label_tensor = label_list[batch_size * i:batch_size * i + batch_size].values
    label_tensor = torch.Tensor(label_tensor)
    label_tensor = label_tensor.long()
    return data_tensor, label_tensor

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return float(num_correct)/float(total)

def train(net,test_data,test_tabel, num_epochs,batch_size,optimizer, criterion,data_path):
    prev_time = datetime.now()

    train_all_num = pd.read_csv(data_path + "/train.csv", sep=',') .iloc[:,0].size # 读取训练数据标签个数
    train_total_batch=train_all_num//batch_size
    test_total_batch=int(len(test_data) / batch_size)
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for i in range(train_total_batch):
            img, label = load_image(i=i, train_test='train', batch_size=batch_size,data_dir=data_path)
            if torch.cuda.is_available():
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            else:
                img = Variable(img)
                label = Variable(label)
            # forward
            output = net(img)
            loss = criterion(output,label)
            _, pred = torch.max(output, 1)  # 返回指定维度 的最大值序号 就是pred
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for i in range(test_total_batch):
            img = test_data[batch_size * i:batch_size * i + batch_size]
            label = test_tabel[batch_size * i:batch_size * i + batch_size]
            if torch.cuda.is_available():
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            else:
                img = Variable(img)
                label = Variable(label)
            output = net(img)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)
        epoch_str = (
            "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
            % (epoch, train_loss / train_total_batch,
               train_acc / train_total_batch, valid_loss / test_total_batch,
               valid_acc / test_total_batch))
        prev_time = cur_time
        print(epoch_str + time_str)
        plt_train_loss.append(train_loss / train_total_batch)
        plt_test_loss.append(valid_loss / test_total_batch)
        plt_train_acc.append(train_acc / train_total_batch)
        plt_test_acc.append(valid_acc / test_total_batch)

def main(data_path, output_path):
    test_data,test_val= load_test_data(data_path)    #测试数据集，测试数据集标签
    epochs = 50        #迭代次数
    batch_size = 20     #一个批次数据
    img_classes = 50   #一共有50个类

    fix_param = False  # 固定参数重新训练????
    # 定义模型
    transfer_model = models.resnet18(pretrained=True)
    if fix_param:
        for param in transfer_model.parameters():
            param.requires_grad = False
    dim_in = transfer_model.fc.in_features  ##### 重写全连接层
    transfer_model.fc = nn.Linear(dim_in, img_classes)  # 增加线性 分类


    transfer_model.conv1 =  nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)   #重写第一层卷积层


    if torch.cuda.is_available():              #    # 使用GPU
        transfer_model = transfer_model.cuda()
    # 定义优化函数和损失函数
    if fix_param:
        optimizer = optim.Adam(transfer_model.fc.parameters(), lr=1e-3)
    else:
        optimizer = optim.Adam(transfer_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(transfer_model,test_data,test_val, epochs,batch_size, optimizer, criterion,data_path)
#画图
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = u'SimHei'  ## 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  ## 解决保存图像是负号'-'显示为方块的问题

    def show_train_history(train_acc, test_acc):
        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    show_train_history(plt_train_acc, plt_test_acc)
    show_train_history(plt_train_loss, plt_test_loss)

    # 测试   #使用的  GPU测试的    如果要提高速度可以尝试修改批次大小
    since = time.time()
    test_total_batch = int(len(test_data) / batch_size)
    valid_loss = 0
    valid_acc = 0
    for i in range(test_total_batch):
        im = test_data[batch_size * i:batch_size * i + batch_size]
        label_cpu = test_val[batch_size * i:batch_size * i + batch_size]
        if torch.cuda.is_available():
            im = Variable(im.cuda())
            label = Variable(label_cpu.cuda())
        else:
            im = Variable(im)
            label = Variable(label_cpu)
        output = transfer_model(im)
        loss = criterion(output, label)
        valid_loss += loss.item()
        valid_acc += get_acc(output, label)
        output_cpu = output.cpu().detach()  # detach就是截断反向传播的梯度流。########################巨大的坑###############
        if i == 0:
            pre_output = output_cpu
            test_label = label_cpu
        else:
            pre_output = torch.cat((pre_output, output_cpu), 0)  # pre_output  预测二维值
            test_label = torch.cat((test_label, label_cpu), 0)
        # print(test_label.type())
        # print(pre_output.type())
        # print(test_label)
        # print(pre_output)

    elips_time = time.time() - since
    print('Acc: {:.4f}, Time: {:.0f}s'.format(valid_acc / test_total_batch, elips_time))
    #torch.save(transfer_model.state_dict(), output_path + '/vgg19.pkl')  # 仅保存和加载模型参数(推荐使用)
    torch.save(transfer_model, output_path+'/FCNmask.pkl')                # 保存和加载整个模型

    ##########绘制roc曲线##############    ##################注意label   pre_out  都在GPU上  使用CPU时需要注意
    # print(test_label.shape)
    test_label = test_label.unsqueeze(1)
    one_hot = torch.zeros(len(test_label), img_classes).scatter_(1, test_label, 1)  # 取到CPU上  如果
    numpy_one_hot = one_hot.numpy()
    numpy_pre_output = pre_output.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(numpy_one_hot.ravel(),
                                             numpy_pre_output.ravel())  # 就是这个东西   ++上后面的画图就行   传入的是NUMPY数据
    auc = metrics.auc(fpr, tpr)
    print('计算auc：', auc)  # ROC曲线围成的面积

    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, color='green', lw=2, label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), color='gray', lw=1, ls='--')
    plt.xlim((-0.01, 1.02))  # 设置图片现在坐标范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')  # 添加网格线
    plt.legend(loc='lower right', fancybox=True, fontsize=12)  # 设置图例的位置右下角
    plt.title(u'鲸鱼尾分类后的ROC和AUC', fontsize=17)
    plt.show()
    ################################################################################

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))                     # 当前目录
    data_path = os.path.join(ROOT_DIR, "data", "data_from_python1")              #数据集路径
    main(data_path= data_path, output_path= data_path)
