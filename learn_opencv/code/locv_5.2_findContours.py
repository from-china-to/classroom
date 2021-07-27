#  图像处理
#! 图像金字塔与轮廓检测

'''
图像轮廓

绘制轮廓
轮廓特征
轮廓近似
边界矩形
外接圆
'''

#? 图像轮廓
'''
cv2.findContours(img,mode,method)
mode:轮廓检索模式

RETR_EXTERNAL ：只检索最外面的轮廓；
RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;        [最常用]
method:轮廓逼近方法

CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
'''

import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 为了更高的准确率，使用二值图像。
img = cv2.imread('./img/contours.png')

#! 第一步 读数据
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#! 第二步 把数据转换成灰度图
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#! 第三步 把图像数据进行二值处理 （通过图像阈值 0 1）
cv_show(thresh,'thresh')

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#! 传入一个二值处理后的图像
#! RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次
#! 以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）
#? binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#? ValueError: not enough values to unpack (expected 3, got 2)
#? 原因是用的是Opencv4.0，把返回值从三个改回两个了
# cv_show(binary,'binary')
# ! 第一个值binary就是刚才的二值图
#np.array(contours).shape
#! 第二个值contours 保存的是轮廓信息，是list结构 需要用np.array转换一下
#! hierarchy是一个层级，结果保存在层级当中
'''
返回值:
　　contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
　　hierarchy:一个ndarray, 元素数量和轮廓数量一样
'''

#? 绘制轮廓

cv_show(img,'img')
#传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy,要不原图会变。。。
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
#! drawContours画轮廓，draw_img原图，contours轮廓是啥，
#! -1 画第几个轮廓?-1是全都画，(0, 0, 255)（BGR），2 线条宽度
cv_show(res,'res')

draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
cv_show(res,'res')

#? 轮廓特征
cnt = contours[0]
#! contours轮廓不能直接拿来计算，所以需要赋值给cnt，这里计算的是第0个轮廓

#面积
cv2.contourArea(cnt)

#周长，True表示闭合的
cv2.arcLength(cnt,True)

#? 轮廓近似
img = cv2.imread('./img/contours2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
cv_show(res,'res')

epsilon = 0.1*cv2.arcLength(cnt,True) 
approx = cv2.approxPolyDP(cnt,epsilon,True)
#! approxPolyDP近似函数，cnt传进来一个轮廓参数， 
#! epsilon传进来一个值进行比较，通常是按照轮廓百分比进行设置的

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show(res,'res')

#? 边界矩形

img = cv2.imread('./img/contours.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#! binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
#! boundingRect矩形，cnt轮廓信息，得到四个点xywh
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#!  (0,255,0)绿色
cv_show(img,'img')

area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
#! 矩形面积
extent = float(area) / rect_area
print ('轮廓面积与边界矩形比',extent)

#? 外接圆

(x,y),radius = cv2.minEnclosingCircle(cnt) 
center = (int(x),int(y)) 
radius = int(radius) 
img = cv2.circle(img,center,radius,(0,255,0),2)
cv_show(img,'img')