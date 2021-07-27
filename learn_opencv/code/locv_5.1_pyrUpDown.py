'''
图像金字塔

高斯金字塔
拉普拉斯金字塔

'''
#  图像处理
#! 图像金字塔与轮廓检测

#? 图像金字塔
'''
高斯金字塔
拉普拉斯金字塔
'''

import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img=cv2.imread("./img/pikachu03.jpg")
cv_show(img,'img')
print (img.shape)

#? 高斯金字塔
# 高斯金字塔：向下采样方法（缩小）
down=cv2.pyrDown(img)
cv_show(down,'down')
print (down.shape)

# 高斯金字塔：向上采样方法（放大）
up=cv2.pyrUp(img)
cv_show(up,'up')
print (up.shape)

up2=cv2.pyrUp(up)
cv_show(up2,'up2')
print (up2.shape)

up=cv2.pyrUp(img)
up_down=cv2.pyrDown(up)
cv_show(up_down,'up_down')
#! 先上采样放大 再下采样缩小，图片两次失真

cv_show(np.hstack((img,up_down)),'up_down')
#! 放在一起 和原图对比
#! np.hstack将参数元组的元素数组按水平方向进行叠加

up=cv2.pyrUp(img)
up_down=cv2.pyrDown(up)
cv_show(img-up_down,'img-up_down')

#? 拉普拉斯金字塔
down=cv2.pyrDown(img)
down_up=cv2.pyrUp(down)
l_1=img-down_up
cv_show(l_1,'l_1')
