'''
版本更新内容：
1.0：   根据Otsu原理 自己写代码 实现Otsu算法（写有大量注释）
1.2：   将自己实现的Otsu算法 封装在myotsu方法中,并且删除大量注释和多于代码
2.0:    对比开闭运算处理的结果，选取开运算消除噪声点
3.0：   用学过的方法实现提取分割区域的轮廓（写有大量注释）
3.2：   删除了大量注释，保留了最精简的代码
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageStat

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def myotsu(imge):
    #? ↓my_otsu算法部分
    rows,cols=np.shape(imge)#!统计出行数列数
    n0=0#前景像素个数    
    n=rows*cols#总像素数
    n1=n-n0#后景像素格式

    #? 二维数组求和
    s=sum(map(sum,imge))#图片总像素值
    u=s/n#图片平均像素值

    mythreshold=0
    my_g=0#存放方差最终结果
    my_th=0#存放阈值最终结果

    #? ↓ for循环部分：求前后景平均像素值 等参数，并计算出最佳阈值
    for i in range(256):

        np.where(imge<mythreshold)#返回坐标，二维图像返回两个数组 一个数组包含x坐标，另一个数组包含y坐标
        n0=np.where(imge<=mythreshold)[0].shape[0]#前景像素点个数    [<=]
        n1=np.where(imge>mythreshold)[0].shape[0]#后景像素点个数     [>]
        if(n0==0 or n1==0): #‘分母’为0的话 直接跳出for，不然最后会得到nan值
            mythreshold +=1
            continue

        w0 = n0/n#前景所占比例
        w1 = n1/n#后景所占比例

        re0, thr0 = cv2.threshold(imge, mythreshold, 255, cv2.THRESH_TOZERO_INV)#前景图片
        #! THRESH_TOZERO_INV  像素值大于阈值时，设置为0，否则不改变
        s0=sum(map(sum,thr0))#前景图片总像素值
        u0=s0/n0#前景图片平均像素值

        re1, thr1 = cv2.threshold(imge, mythreshold, 255, cv2.THRESH_TOZERO)#后景图片
        #! THRESH_TOZERO 像素值大于阈值时，不变，否则取0
        s1=sum(map(sum,thr1))#背景图片总像素值
        u1=s1/n1#背景图片平均像素值

        g=w0*w1*(u0*u0-2*u0*u1+u1*u1)#综合计算 前景u0和背景u1的方差g，代码是g公式化简后的结果

        if g>my_g:
            my_g=g
            my_th=mythreshold

        mythreshold +=1
    print("最优阈值："+str(my_th))

    my_ret, my_th = cv2.threshold(img, my_th, 255, cv2.THRESH_BINARY)
    
    #? ↑my_otsu算法部分
    return my_ret,my_th

img = cv2.imread('./img/whale.png', 0)

# 我的otsu 阈值法
ret1, th1 = myotsu(img)
print("MyOtsu 得到的阈值：\t"+str(ret1))

# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

images = [img, 0, th1, img, 0, th2]
titles = ['Original', 'Histogram', 'my_Otsu:'+str(ret1),
         'Original', 'Histogram', "Otsu's"+str(ret2),]

for i in range(2):
    # 绘制原图
    plt.subplot(2, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制直方图plt.hist, ravel函数将数组降成一维
    plt.subplot(2, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制阈值图
    plt.subplot(2, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()

#! 开运算与闭运算
kernel = np.ones((5,5),np.uint8) 
o_img = th1.copy()# 开：先腐蚀，再膨胀
opening = cv2.morphologyEx(o_img, cv2.MORPH_OPEN, kernel)

c_img = th1.copy()# 闭：先膨胀，再腐蚀
closing = cv2.morphologyEx(c_img, cv2.MORPH_CLOSE, kernel)

plt.figure()
plt.subplot(2,2,1)		# 将画板分为2行两列，本幅图位于第一个位置
plt.imshow(img,cmap="gray")
plt.subplot(2,2,2)		# 将画板分为2行两列，本幅图位于第二个位置
plt.imshow(th1,cmap="gray")
plt.subplot(2,2,3)		# 将画板分为2行两列，本幅图位于第3个位置
plt.imshow(opening,cmap="gray")
plt.subplot(2,2,4)		# 将画板分为2行两列，本幅图位于第3个位置
plt.imshow(closing,cmap="gray")
plt.show()





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

# 为了更高的准确率，使用二值图像。
#! 第一步 读数据
#! 第二步 把数据转换成灰度图
#! 第三步 把图像数据进行二值处理 （通过图像阈值 0 1）
cv_show(opening,'thresh')
print(opening)
cv2.imwrite('./img/whale_otsued.png',opening)

contours,hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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

cv_show(opening,'img')
#传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
# 注意需要copy,要不原图会变。。。
draw_img = opening.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
#! drawContours画轮廓，draw_img原图，contours轮廓是啥，
#! -1 画第几个轮廓?-1是全都画，(0, 0, 255)（BGR），2 线条宽度
cv_show(res,'res')

draw_img = opening.copy()
res = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)
cv_show(res,'res')

#? 轮廓近似

ret, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = opening.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
cv_show(res,'res')

epsilon = 0.1*cv2.arcLength(cnt,True) 
approx = cv2.approxPolyDP(cnt,epsilon,True)
#! approxPolyDP近似函数，cnt传进来一个轮廓参数， 
#! epsilon传进来一个值进行比较，通常是按照轮廓百分比进行设置的

draw_img = opening.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv_show(res,'res')
