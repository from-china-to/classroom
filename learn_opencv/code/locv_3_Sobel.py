'''
图像梯度处理

图像梯度-Sobel算子Scharr算子Laplacian 算子
'''
#  图像处理
#! 图像梯度处理

#? 图像梯度-Sobel算子

import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB

img = cv2.imread('./img/pie.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
#! 梯度 相当于边缘检测，不同主体之间的边缘。
#! 计算某一点左边是什么 右边是什么 看左右两边一样不一样

#! 形态学梯度：膨胀减去腐蚀
#! 图像梯度：指边缘处产生的，常指二值图像中的黑白处。

'''
dst = cv2.Sobel(src, ddepth, dx, dy, ksize)

ddepth:图像的深度
dx和dy分别表示水平和竖直方向
ksize是Sobel算子的大小
'''
#? st = cv2.Sobel(src, ddepth, dx, dy, ksize)
#! -----------白减去黑，或者黑减去白
#! ddepth:图像的深度，一般会写CV_64F，可以出现负数
#! dx和dy分别表示水平和竖直方向，水平：1,0（右边减左边）；竖直：0,1（下面减上面）；
#! ksize是Sobel算子的大小

'''
原理：
梯度本质上就是导数。OpenCV 提供了三种不同的梯度滤波器，或者说高通滤波器：
Sobel，Scharr 和Laplacian。Sobel，
Scharr 其实就是求一阶或二阶导数。
Scharr 是对Sobel（使用小的卷积核求解求解梯度角度时）的优化。
Laplacian 是求二阶导数
https://blog.csdn.net/weixin_46318945/article/details/107710008
'''

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
cv_show(sobelx,'sobelx')

# 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
#! 白到黑是正数，黑到白就是负数了，所有的负数opencv默认会被截断成0，所以要取绝对值
#! （若不取绝对值，那么黑到白会是负数，opencv默认处理为0，图像此处边缘就无法显示。只显示一边的边缘，如上图）
#! https://blog.csdn.net/xiachong27/article/details/88371771

#? 分开计算再相加
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
#! -------------取绝对值
cv_show(sobelx,'sobelx')

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)  
cv_show(sobely,'sobely')

# 分别计算x和y，再求和

sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv_show(sobelxy,'sobelxy')

'''
Sobel 算子和Scharr 算子
Sobel 算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好。
你可以设定求导的方向（xorder 或yorder）。还可以设定使用的卷积核的大小（ksize）。
如果ksize=-1，会使用3x3 的Scharr 滤波器，它的的效果要比3x3 的Sobel 滤波器好
（而且速度相同，所以在使用3x3 滤波器时应该尽量使用Scharr 滤波器）
'''
'''
在上面这个例子的注释有提到：当我们可以通过参数-1 来设定输出图像的深度（数据类型）
与原图像保持一致，但是我们在代码中使用的却是cv2.CV_64F。
这是为什么呢？想象一下一个从黑到白的边界的导数是整数，而一个从白到黑的边界点导数却是负数。
如果原图像的深度是np.int8 时，所有的负值都会被截断变成0，换句话说就是把把边界丢失掉。
所以如果这两种边界你都想检测到，最好的的办法就是将输出的数据类型设置的更高，
比如cv2.CV_16S，cv2.CV_64F 等。取绝对值然后再把它转回到cv2.CV_8U。
原文链接：https://blog.csdn.net/weixin_46318945/article/details/107710008
'''

# 不建议直接计算
#! 如果像下面直接计算（直接写1,1），产生的梯度会不太连续，不如分开再加和效果好。（如下图）


#? 直接计算水平与竖直方向
sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy) 
cv_show(sobelxy,'sobelxy')

img = cv2.imread('./img/lena.jpg',cv2.IMREAD_GRAYSCALE)
cv_show(img,'img')

img = cv2.imread('./img/lena.jpg',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv_show(sobelxy,'sobelxy')

img = cv2.imread('./img/lena.jpg',cv2.IMREAD_GRAYSCALE)
sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy) 
cv_show(sobelxy,'sobelxy')

'''
Laplacian 算子
拉普拉斯算子可以使用二阶导数的形式定义，可假设其离散实现类似于二阶Sobel 导数，
事实上，OpenCV 在计算拉普拉斯算子时直接调用Sobel 算子。
'''

#不同算子的差异
img = cv2.imread('./img/lena.jpg',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)   
sobely = cv2.convertScaleAbs(sobely)  
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)  

scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)   
scharry = cv2.convertScaleAbs(scharry)  
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0) 

laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)   

res = np.hstack((sobelxy,scharrxy,laplacian))
cv_show(res,'res')

img = cv2.imread('./img/lena.jpg',cv2.IMREAD_GRAYSCALE)
cv_show(img,'img')