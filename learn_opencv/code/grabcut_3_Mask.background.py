'''
Grabcut应用背景替换
概述
✔️ 使用Grabcut实现图像对象提取，通过背景图像替换，实现图像合成，通过对背景图像高斯模糊实现背景虚化效果，完整的步骤如下：

1·ROI区域选择；
2·Grabcut对象分割；
3·Mask生成，并转化为alpha值；
4·使用 com = alpha*fg + (1-alpha)*bg 公式融合图片。
'''

import cv2 as cv
import numpy as np

src = cv.imread("./img/pikachu03.jpg")
src = cv.resize(src, (0,0), fx=0.5, fy=0.5) #? 缩小或者放大函数至某一个大小
r = cv.selectROI('input', src, False)  #? 在一幅图像中，如何选择自己感兴趣的区域
# 返回 (x_min, y_min, w, h)

# roi区域
roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#? 注意：这里image[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 需要横纵坐标调换一下
#! ( y_min : y_min + h , x_min  : x_min + w)

img = src.copy()
cv.rectangle(img, (int(r[0]), int(r[1])),(int(r[0])+int(r[2]), int(r[1])+ int(r[3])), (255, 0, 0), 2)
#? void cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int line_type=8, int shift=0 );
# img，图像；pt1，矩形的一个顶点；pt2，矩形对角线上的另一个顶点；
# color，线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）；
# thickness，组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形；
# line_type，线条的类型（见cvLine的描述）；shift，坐标点的小数点位数。
#? 函数 cvRectangle 通过对角线上的两个顶点绘制矩
# 注：cvRectangle函数只是在img图像的指定区域画了一个指定大小，边框颜色，边框粗细的矩形框而已，并不会自动显示
#! 找到一个详细的解释 https://blog.csdn.net/sinat_41104353/article/details/85171185

# 原图mask
mask = np.zeros(src.shape[:2], dtype=np.uint8)
#? img.shape[:2]取彩色图片的长、宽

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组  13 * iterCount
fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组  13 * iterCount

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)
#有时候 报错 cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)
#? iterCount:指定迭代次数11
#? mode:有三个值可用
#           cv::GC_INIT_WITH_RECT//用矩阵初始化grabCut
#           cv::GC_INIT_WITH_MASK//用掩码初始化grabCut
#           cv::GC_EVAL//执行分割

# 提取前景和可能的前景区域
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
#? np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y。
#? 就是转换numpy数组的数据类型
# 用法：arr.astype(“具体的数据类型”)
background = cv.imread("./img/pikachu03.jpg")

h, w, ch = src.shape    #? 默认读入三通道图片
# 参数-1为按原通道读入，不写的话默认读入三通道图片，例如（112，112，3）

background = cv.resize(background, (w, h))
cv.imwrite("background.jpg", background)

mask = np.zeros(src.shape[:2], dtype=np.uint8)
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel,5,mode=cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

# 高斯模糊
se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#? getStructuringElement函数会返回指定形状和尺寸的结构元素。
# 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
# 矩形：MORPH_RECT;
# 交叉形：MORPH_CROSS;
# 椭圆形：MORPH_ELLIPSE

cv.dilate(mask2, se, mask2)
#? 膨胀
# 取核中像素值的最大值代替锚点位置的像素值，这样会使图像中较亮的区域增大，较暗的区域减小。
# 如果是一张黑底，白色前景的二值图，就会使白色的前景物体颜色面积变大，就像膨胀了一样
# dst = cv2.dilate(src,kernel,anchor,iterations,borderType,borderValue)
#         src: 输入图像对象矩阵,为二值化图像
#         kernel:进行腐蚀操作的核，可以通过函数getStructuringElement()获得
#         anchor:锚点，默认为(-1,-1)
#         iterations:腐蚀操作的次数，默认为1
#         borderType: 边界种类
#         borderValue:边界值
mask2 = cv.GaussianBlur(mask2, (5, 5), 0)
cv.imshow('background-mask',mask2)
cv.imwrite('background-mask.jpg',mask2)


# 虚化背景
background = cv.GaussianBlur(background, (0, 0), 15)
#? 此函数利用高斯滤波器平滑一张图像。该函数将源图像与指定的高斯核进行卷积。
# GaussianBlur(src,ksize,sigmaX,dst= None,sigmaY= None,borderType= None)
# src:输入图像
# ksize:(核的宽度,核的高度)，输入高斯核的尺寸，核的宽高都必须是正奇数。否则，将会从参数sigma中计算得到。
# dst:输出图像，尺寸与输入图像一致。
# sigmaX:高斯核在X方向上的标准差。
# sigmaY:高斯核在Y方向上的标准差。默认为None，如果sigmaY=0，则它将被设置为与sigmaX相等的值。如果这两者都为0,则它们的值会从ksize中计算得到。计算公式为：
# borderType:像素外推法，默认为None

mask2 = mask2/255.0
a =  mask2[..., None]
#? mask2[..., None] == mask2[:, None]
# ... : 原序列不变， none相当于多加了一个通道。把 原来二维的 变成了 二维加一个通道

# 融合方法 com = a*fg + (1-a)*bg
result = a* (src.astype(np.float32)) +(1 - a) * (background.astype(np.float32))


cv.imshow("result", result.astype(np.uint8))
cv.imwrite("result.jpg", result.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()