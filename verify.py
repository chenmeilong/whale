#batsize=10  43s
#####
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import cv2
import os
import use_fcn
mpl.rcParams['font.sans-serif'] = u'SimHei'  ## 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  ## 解决保存图像是负号'-'显示为方块的问题
batch_size = 10  # 一个批次数据
img_classes = 50  # 一共有50个类
test_dir="./data/test"
model_dir="./data\model/FCNmask.pkl"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_test_data(data_path):    #加载全部的测试数据集 ，这样可以提高运行效率  ，因为训练数据集需要数据增强，所以不方便这样加载
    csv_test_data = pd.read_csv(data_path+ "/test.csv", sep=',')  # 读取测试数据标签
    image_test_list = csv_test_data.iloc[:, 0].values
    label_test_list=csv_test_data.iloc[:, 1].values
    label_list=[]

    for i in range(csv_test_data.iloc[:, 0].size):
        im =cv2.imread(data_path+"/"+"test"+"/"+image_test_list[i])  # 打开图片
        if os.path.exists(data_path+"/"+"test"+"/"+image_test_list[i]):
            imge = transform(im)
            imge=use_fcn.pre_fcn(imge)    # ###################加入fcn            mask
            imge = torch.unsqueeze(imge, 0)
            label_list.append(label_test_list[i])
            if i == 0:
                test_tensor = imge
            else:
                test_tensor = torch.cat([test_tensor, imge], 0)
    label_test_tensor = torch.Tensor(label_list)
    label_test_tensor = label_test_tensor.long()
    print("成功加载测试数据集：",test_tensor.shape,label_test_tensor.shape)
    return test_tensor,label_test_tensor   #训练数据集标签，测试数据集标签      #全部存为tensor

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return float(num_correct)/float(total)

test_tensor,label_tensor_tensor=load_test_data(test_dir)

#测试   #使用的  GPU测试的    如果要提高速度可以尝试修改批次大小
since = time.time()
test_total_batch=int(len(test_tensor) / batch_size)
valid_acc = 0

im = Variable(test_tensor)
Resnet_model =torch.load(model_dir,map_location=lambda storage, loc: storage)  #GPUc参数加载到CPU中
Resnet_model.eval()

for i in range(test_total_batch):
    im = test_tensor[batch_size * i:batch_size * i + batch_size]
    label_cpu = label_tensor_tensor[batch_size * i:batch_size * i + batch_size]
    im = Variable(im)
    label = Variable(label_cpu)
    output= Resnet_model(im)
    valid_acc += get_acc(output, label)
    output_cpu = output.detach()    #detach就是截断反向传播的梯度流。########################巨大的坑###############
    if i == 0:
        pre_output=output_cpu
    else:
        pre_output=torch.cat((pre_output,output_cpu),0)    #pre_output  预测二维值
    # print(test_label.type())
    # print(pre_output.type())
    # print(test_label)
    # print(pre_output)

if len(test_tensor) % batch_size!=0:
    im = test_tensor[batch_size * test_total_batch:len(test_tensor)]
    label_cpu = label_tensor_tensor[batch_size * test_total_batch:len(test_tensor)]
    im = Variable(im)
    label = Variable(label_cpu)
    output = Resnet_model(im)
    valid_acc += get_acc(output, label)
    output_cpu = output.detach()
    pre_output = torch.cat((pre_output, output_cpu), 0)  # pre_output  预测二维值
    test_total_batch=test_total_batch+1

elips_time = time.time() - since
print('Acc: {:.4f}, Time: {:.0f}s'.format(valid_acc / test_total_batch, elips_time))

##########绘制roc曲线##############    ##################注意label   pre_out  都在GPU上  使用CPU时需要注意
test_label = label_tensor_tensor.unsqueeze(1)   #增加维度
one_hot = torch.zeros(len(test_label),img_classes).scatter_(1,test_label, 1)    #取到CPU上  如果
numpy_one_hot = one_hot.numpy()
numpy_pre_output = pre_output.numpy()
print("模型输出的shape",numpy_pre_output.shape)
fpr, tpr, thresholds = metrics.roc_curve(numpy_one_hot.ravel(), numpy_pre_output.ravel())  #就是这个东西   ++上后面的画图就行   传入的是NUMPY数据
auc = metrics.auc(fpr, tpr)
print( '计算auc：', auc)               #ROC曲线围成的面积

# FPR就是横坐标,TPR就是纵坐标
plt.plot(fpr, tpr, color='green', lw=2, label=u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), color='gray', lw=1, ls='--')
plt.xlim((-0.01, 1.02))   #设置图片现在坐标范围
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))   #设置刻度
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')   #添加网格线
plt.legend(loc='lower right', fancybox=True, fontsize=12)  #设置图例的位置右下角
plt.title(u'鲸鱼尾分类后的ROC和AUC', fontsize=17)
plt.show()
