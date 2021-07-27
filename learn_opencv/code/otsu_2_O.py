#https://blog.csdn.net/u010128736/article/details/52801310

'''
一、OTSU法（大津阈值分割法）介绍
  OTSU算法是由日本学者OTSU于1979年提出的一种对图像进行二值化的高效算法，
是一种自适应的阈值确定的方法，又称大津阈值分割法，是最小二乘法意义下的最优分割。
'''
'''
二、单阈值OTSU法
  设图像包含L个灰度级，灰度值为i的像素点个数为Ni，像素总点数为：
N=N0+N1+⋯+NL−1
则灰度值为i的点的概率为：
Pi=Ni/N
... ...

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./img/whale.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#! cv2.cvtColor()方法用于将图像从一种颜色空间转换为另一种颜色空间。
#! 用法： cv2.cvtColor(src, code[, dst[, dstCn]])
#! 参数：
# src:它是要更改其色彩空间的图像。
# code:它是色彩空间转换代码。
# dst:它是与src图像大小和深度相同的输出图像。它是一个可选参数。
# dstCn:它是目标图像中的频道数。如果参数为0，则通道数自动从src和代码得出。它是一个可选参数。
#! 返回值：它返回一个图像。

plt.subplot(131), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.hist(image.ravel(), 256)
#! ravel()：如果没有必要，不会产生源数据的副本
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
plt.subplot(133), plt.imshow(th1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
plt.show()