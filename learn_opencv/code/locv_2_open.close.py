'''
图像形态学处理
形态学-腐蚀操作-膨胀操作
开运算与闭运算
梯度运算
礼帽与黑帽

开：先腐蚀，再膨胀
闭：先膨胀，再腐蚀

梯度=膨胀-腐蚀

礼帽 = 原始输入-开运算结果
黑帽 = 闭运算-原始输入
'''
#  图像处理
#! 图像形态学处理

#? 形态学-腐蚀操作
import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB

img = cv2.imread('./img/cc.png')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 腐蚀操作：通常对二值图象数据进行腐蚀操作


kernel = np.ones((3,3),np.uint8) 
erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imshow('erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 腐蚀后 杂纹消失 字体线条变细


pie = cv2.imread('./img/pie.png')

cv2.imshow('pie', pie)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((30,30),np.uint8) 
erosion_1 = cv2.erode(pie,kernel,iterations = 1)
erosion_2 = cv2.erode(pie,kernel,iterations = 2)
erosion_3 = cv2.erode(pie,kernel,iterations = 3)
res = np.hstack((erosion_1,erosion_2,erosion_3))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 一个圆，腐蚀的次数越多，变得越小


#? 形态学-膨胀操作
img = cv2.imread('./img/cc.png')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 原始图像中有杂纹

kernel = np.ones((3,3),np.uint8) 
dige_erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imshow('erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 先腐蚀操作 去除杂文，同时原始图像也被迫发生改变

kernel = np.ones((3,3),np.uint8) 
dige_dilate = cv2.dilate(dige_erosion,kernel,iterations = 1)

cv2.imshow('dilate', dige_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 再膨胀操作 还原原始图像，达到驱逐杂纹的效果

pie = cv2.imread('./img/pie.png')

kernel = np.ones((30,30),np.uint8) 
dilate_1 = cv2.dilate(pie,kernel,iterations = 1)
dilate_2 = cv2.dilate(pie,kernel,iterations = 2)
dilate_3 = cv2.dilate(pie,kernel,iterations = 3)
res = np.hstack((dilate_1,dilate_2,dilate_3))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
#! 一个圆 膨胀数值越大 越方


#! 开运算与闭运算

# 开：先腐蚀，再膨胀
#! 祛除笔迹杂痕
img = cv2.imread('./img/cc.png')

kernel = np.ones((5,5),np.uint8) 
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 闭：先膨胀，再腐蚀
#! 先把杂痕放大了，杂痕大了更加腐蚀不掉了
img = cv2.imread('./img/cc.png')

kernel = np.ones((5,5),np.uint8) 
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

#? 梯度运算

# 梯度=膨胀-腐蚀
#! 膨胀比原来胖一圈，腐蚀比原来瘦一圈
pie = cv2.imread('./img/pie.png')
kernel = np.ones((7,7),np.uint8) 
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode(pie,kernel,iterations = 5)

res = np.hstack((dilate,erosion))

cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
#! morphology形态学；MORPH_GRADIENT梯度运算

cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

#? 礼帽与黑帽
# 礼帽 = 原始输入-开运算结果
# 黑帽 = 闭运算-原始输入

#礼帽
#! 原始输入 - 去除杂痕后的图案 = 杂痕
img = cv2.imread('./img/cc.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#! MORPH_TOPHAT 礼帽顶帽tophat
cv2.imshow('tophat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

#黑帽
#! 杂痕变大后的图案 - 原始图案 = 原始图案的轮廓
img = cv2.imread('./img/cc.png')
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat ', blackhat )
cv2.waitKey(0)
cv2.destroyAllWindows()