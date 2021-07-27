'''
2021年7月26日
功能1：均值漂移处理图像
功能2：截取图片中矩形区域
'''

#? OpenCV图像处理-均值漂移&Grabcut分割
#! https://zhuanlan.zhihu.com/p/85813603

import cv2 as cv
import numpy as np

src = cv.imread("./img/pikachu03.jpg")
dst = cv.pyrMeanShiftFiltering(src, 25, 40, None, 2) #? 图像均值漂移
cv.imshow("result", np.hstack((src,dst)))
cv.waitKey()
cv.destroyAllWindows()
'''
图像均值漂移
✔️ MeanShfit 均值漂移算法是一种通用的聚类算法，通常可以实现彩色图像分割。

dst = cv.pyrMeanShiftFiltering(src, sp, sr, maxLevel, termcrit)
其中： - src --> 输入图像; - dst --> 输出结果; - sp --> 表示空间窗口大小;
 - sr --> 表示表示颜色空间; - maxLevel --> 表示金字塔层数，总层数为maxlevel+1; 
 - termcrit --> 表示停止条件;
'''




'''
Grabcut图像分割

✔️ Grabcut是基于图割(graph cut)实现的图像分割算法，它需要用户输入一个bounding box
作为分割目标位置，实现对目标与背景的分离/分割。
✔️ Grabcut分割速度快，效果好，支持交互操作，因此在很多APP图像分割/背景虚化的软件中经常使用。

cv2.grabCut(img, rect, mask,
            bgdModel, fgdModel, 
            iterCount, mode = GC_EVAL)

img --> 输入的三通道图像；
mask --> 输入的单通道图像，初始化方式为GC_INIT_WITH_RECT表示ROI区域可以被初始化为：
GC_BGD --> 定义为明显的背景像素 0
GC_FGD --> 定义为明显的前景像素 1
GC_PR_BGD --> 定义为可能的背景像素 2
GC_PR_FGD --> 定义为可能的前景像素 3
rect --> 表示roi区域；
bgdModel --> 表示临时背景模型数组；
fgdModel --> 表示临时前景模型数组；
iterCount --> 表示图割算法迭代次数, 次数越多，效果越好；
mode --> 当使用用户提供的roi时候使用GC_INIT_WITH_RECT。
'''
'''
selectROI(windowName, img, showCrosshair=None, fromCenter=None):
    .   参数windowName：选择的区域被显示在的窗口的名字
    .   参数img：要在什么图片上选择ROI
    .   参数showCrosshair：是否在矩形框里画十字线.
    .   参数fromCenter：是否是从矩形框的中心开始画
返回的是一个元组[min_x,min_y,w,h]：
第一个值为矩形框中最小的x值
第二个值为矩形框中最小的y值
第三个值为这个矩形框的宽
第四个值为这个矩形框的高
原文链接：https://blog.csdn.net/fjswcjswzy/article/details/105881899
'''
import cv2 as cv
import numpy as np

src = cv.imread("./img/pikachu03.jpg")
src = cv.resize(src, (0,0), fx=0.5, fy=0.5) #? 缩小或者放大函数至某一个大小
r = cv.selectROI('input', src, False)       #? 在一幅图像中，如何选择自己感兴趣的区域
# 返回 (x_min, y_min, w, h)

# roi区域
roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#? 注意：这里image[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])] 需要横纵坐标调换一下
#! ( y_min : y_min + h , x_min  : x_min + w)

# 原图mask
mask = np.zeros(src.shape[:2], dtype=np.uint8)
'''
img.shape[:2]取彩色图片的长、宽
img.shape[:3]取彩色图片的长、宽、通道
img.shape[0]图像的垂直尺寸（高度）
img.shape[1]图像的水平尺寸（宽度）
img.shape[2]图像的通道数
注：矩阵中，[0]表示行数，[1]表示列数
'''

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组
#! img.shape[:2]取彩色图片的长、宽(1,65)
fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)
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

print(mask2.shape)

result = cv.bitwise_and(src,src,mask=mask2)
#? cv2.bitwise_and()是对二进制数据进行“与”操作，
# 即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，
# 1&1=1，1&0=0，0&1=0，0&0=0
# OutputArray dst  = cv2.bitwise_and(InputArray src1, InputArray src2, InputArray mask=noArray());//dst = src1 & src2
#! 利用掩膜（mask）进行“与”操作，
# 即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已

cv.imwrite('result.jpg', result)
cv.imwrite('roi.jpg', roi)

cv.imshow('roi', roi)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()

'''
输入：

采用 selectROI, 可以在图中自己选定ROI区域：
选定后，按enter 或则 Space 进行grabcut；
重新选ROI，只需用鼠标重新选择即可；
按 c 结束程序。
'''



