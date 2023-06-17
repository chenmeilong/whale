import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import cv2
import os

fcn_model = torch.load('checkpoints/model100.pkl')
fcn_model.eval()
# 数据增强
data_transforms = {
    'test':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
}
def pre_fcn(img_tensor):
    #这里需要resize   tensor   成指定大小  160*160         print(img_tensor.shape)
    numpy_tensor = img_tensor.numpy()
    numpy_tran = np.transpose(numpy_tensor, (1, 2, 0))  # 图片转置
    # cv2.imshow('img_1', numpy_tran)
    # cv2.waitKey()
    numpy_resize=cv2.resize(numpy_tran, (160, 160))
    #print(cv_img.shape)
    numpy_tran = np.transpose(numpy_resize, (2, 0, 1))  # 图片转置
    tensor_numpy= torch.Tensor(numpy_tran)
    img = tensor_numpy.unsqueeze(0)  ##torch.Size([1, 3, 160, 160]),增加一个维度
    if torch.cuda.is_available():
        im = Variable(img.cuda())
    pre = fcn_model(im)
    numpy_array = pre.cpu().detach().numpy()
    numpy_array = np.squeeze(numpy_array, axis=0)
    output_np = np.argmin(numpy_array, axis=0)  # 输出最小值所在的位置
    # print (output_np .shape)     #形状 160*160
    img1 = output_np.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)  # 膨胀腐蚀
    img1.dtype = "int8"
    img1 = abs(img1 - 1)  # 图片颜色翻转   颜色fanzhaun：  0-1  1-0
    img1.dtype = "uint8"
    # array = img1.tolist()
    # print(array)
    img1 = cv2.resize(img1,(img_tensor.shape[1],img_tensor.shape[2]),cv2.INTER_CUBIC)  #放大
    # from matplotlib import pyplot as plt     #显示fcn图片
    # plt.imshow(img1)
    # plt.show()
    np_fcn = np.array(img1)
    np_fcn = np_fcn[np.newaxis, :]
    tensor_fcn = torch.Tensor(np_fcn)
    out_tensor = torch.cat((img_tensor, tensor_fcn), dim=0)  # print(out_tensor.shape)
    return out_tensor

if __name__ == '__main__':
    dir_data = "test/1.jpg"
    img = cv2.imread(dir_data)
    print(img.shape)
    img = data_transforms["test"](img)
    concat_tensor = pre_fcn(img)
    print(concat_tensor.shape)
    #print(concat_tensor)
    # 返回一个tensor 【4，160，160】



