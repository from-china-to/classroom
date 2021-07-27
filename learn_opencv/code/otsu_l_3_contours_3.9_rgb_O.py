
#? https://blog.csdn.net/qq_27261889/article/details/80659285

#! usr/bin/env python
# coding:utf-8
# for opencv
 
# 2018年6月11日21:23:18
# 参考网址：https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#basic-ops
# 目标：
# 1、访问像素并修改像素
# 2、图像属性
# 3、通过像素的切片操作设置感兴趣区域（ROI，region of interest）
# 4、分离融合图像的通道
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
###########目标1：访问并修改像素###########
img = cv2.imread('./img/whale.png')
plt.imshow(img)
plt.show()
px = img[100][100]
print(px)
# 利用行列坐标来读取图像的像素，单通道图就是一个值，BGR就是三个值
# （注意，opencv中图像的通道是BGR的顺序）
print(img[100,100,0])
# 通过第三个下表来单独访问一个通道值
 
img[100,100] = [21, 0, 120]
# 通过下标来修改单个像素值
 
# numpy更加快，我们经常使用numpy来修改像素值
# 使用item()和itemset()来访问和设置像素值
print(img.item(100,100,2))
# 可以利用item函数来访问单个通道的值。
img.itemset((100,100,2),20)
print(img.item(100,100,2))
 
###########目标1：总结###########
# 1、可以利用下标索引来访问像素值，主要有三个下标，像素的二维索引和通道索引，注意通道索引是BGR的顺序
# 2、最好利用numpy中item和itemset来访问像素值，因为速度比较快
 
 
 
################目标2：图像大小等属性#########
print(img.shape)
# 利用shape来获得图像的大小和通道数
print(img.size)
# 利用size方法来获得图像的所有像素点的个数，包括通道数
print(img.dtype)
# 利用dtype来获得data的type（），有uint8等
# dtype是一个比较重要的东西，经常会因为dtype出错
 
################目标2：总结#########
# 1、注意图像的编码方式dtype，如果超出范围就容易出错了。
 
 
 
################目标3：利用切片设置感兴趣区域#########
# 选择图像的某一部分进行研究，可以利用numpy进行索引
# 以下先选择一个区域，并复制到图像的另一个区域
foot = img[10:20,30:40]
# 通过numpy矩阵的切片操作来选择图像的某一个区域
img[70:80,100:110] = foot
# 将ROI区域的一个部分放在图像的另一个区域
cv2.imshow('img',img)
# 展示图片
 
 
################目标4：分离并融合图像#########
# 由于cv2.imread得到的图像是按照BGR进行排列的，而我们经常需要RGB的图像，因此需要先分离图像通道，然后再融合图像
b,g,r = cv2.split(img)
# 通过split来分离通道
b0 = img[:,:,0]
# 也可以通过切片操作来进行得到
# 注意split是一个比较耗时的操作，尽量不要使用
img = cv2.merge((r,g,b))
cv2.imshow('img1',img)
cv2.waitKey()
cv2.destroyAllWindows()
# 记住需要这个方法来销毁所有的窗口，否则会吃内存