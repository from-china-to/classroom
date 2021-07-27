'''
图像 阈值与平滑处理
HSV
均值滤波（最基本的滤波操作）
方框滤波
高斯滤波
! 绘制一个高斯曲线
! 绘制一个3D高斯曲线 
二维面振幅分布图
三维曲面振幅分布图
中值滤波
'''

#  图像处理
#! 阈值与平滑处理

# 灰度图
import cv2 #opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB
#! %matplotlib inline 

img=cv2.imread('./img/pikachu01.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray.shape

print(img_gray.shape)

cv2.imshow("img_gray", img_gray)
cv2.waitKey(0)    
cv2.destroyAllWindows() 

'''
HSV
H - 色调（主波长）。
S - 饱和度（纯度/颜色的阴影）。
V值（强度）
'''
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.imshow("hsv", hsv)
cv2.waitKey(0)    
cv2.destroyAllWindows()

'''
图像阈值
ret, dst = cv2.threshold(src, thresh, maxval, type)
【src:原始图像，通过opencv拖进来图像就可以 【thresh：指定阈值是多少，不能是0.7 0.8这样的百分比，通常比较常见的用127，因为我们的范围是0-255 【msxval：最大的值 就是 255 【type:最重要的是这个参数，表示要做阈值这么一个事情 要选择的功能或者方法是什么。怎么判断阈值，判断后怎么处理 都是由type来决定的。】
src： 输入图，只能输入单通道图像，通常来说为灰度图
dst： 输出图
thresh： 阈值
maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0 【二值法，大于阈值取一个值，小于阈值取一个值。】
cv2.THRESH_BINARY_INV THRESH_BINARY的反转 【inv-将前面的方法进行一个反转】
cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变 【设置一个截断值，eg 截断值127，则 >127部分 就 =127】
cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.T
'''
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
#! 亮的全变为白，暗的全变成黑
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
#! 和上个图相比，黑变白 白变黑
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
#! 亮的更亮，暗的不变
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
#! 暗的更暗，亮的不变
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
#! 和上个图相比，黑变白 白变黑

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# cv2.imshow("hsv", image[0])
#? 这里CV2调用image[0] 也就是img时 一切正常
# plt.imshow(images[0],'gray')
# plt.show()
#? 这里plt再次调用img图片 并gray后，皮卡丘会变‘蓝’

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 图像平滑

img = cv2.imread('./img/pikachu04n.png')

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 均值滤波（最基本的滤波操作）
#! 简单的平均卷积操作
'''
假如拿到了一个图像，来看像素点的矩阵；
然后指定一个 核 的大小 eg 【3 * 3】
对【33】内做平滑处理，其实就是对里面每个像素点进行平滑处理
eg 对最中心的像素点‘204’进行变换，现在我们要进行的是‘均值滤波’
【均值：=（ 121+75+78+。。。+235 ）/9】
通常 我们要先构造一个 3 *3的卷积矩阵（卷积核）：
【1，1，1
  1，1，1
  1，1，1】
  然后求内积，1*121 1*75 1*78 .。。 1*235 最后在求一个平均
  
'''
blur = cv2.blur(img, (3, 3))
#!调用CV2均值滤波操作，输入图像数据img，filter（核）的大小3*3（或者5*5通常是奇数）

cv2.imshow('blur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 方框滤波
#! 基本和均值一样，可以选择归一化
'''
和 均值滤波很相似，
‘-1’：python中指定一个-1是说的想让他自己自动地进行计算，
表示我们所得到的结果和原始数据的颜色通道数是一样的，就是在原始通道上进行计算
-1表示一致，通常是固定的参数-1
我们在均值滤波中3*3的所有位置加起来人后/9 像不像一个归一化的操作
normalize 归一化，如果做归一化=True 就和均值滤波完全一样了
'''
box = cv2.boxFilter(img,-1,(3,3), normalize=True)  

cv2.imshow('box', box)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 方框滤波
#! 基本和均值一样，可以选择归一化,容易越界
'''
 （eg 75+204 >255）
 越界部分全部取值 255，255白 很白
'''
box = cv2.boxFilter(img,-1,(3,3), normalize=False)  

cv2.imshow('box', box)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
'''
先看一下高斯函数的图像，高斯函数是说 越接近均值，可能性越大
越接近x=0，y值就越大
均值滤波中 所有值加一起/9，这样公平吗？
拿204当作中心点，75距离204比较近的，78距离204比较远
按照高斯函数的思想，据离我越近的跟我关系越紧密 应该越重视，
所以我们的filter可以这么设置：（自己构造一个权重矩阵）
【0.6  0.8  0.6
  0.8  1    0.8
  0.6  0.8  0.6】
  有一个远近的关系，距离越近发挥效果越强，距离越远发挥效果就没那么大了

'''
aussian = cv2.GaussianBlur(img, (5, 5), 1)  

cv2.imshow('aussian', aussian)
cv2.waitKey(0)
cv2.destroyAllWindows()

#! 绘制一个高斯曲线 便于理解
import numpy as np
import matplotlib.pyplot as plt
import math
 
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
 
mean1, sigma1 = 0, 1
x1 = np.linspace(mean1 - 6*sigma1, mean1 + 6*sigma1, 100)
 
mean2, sigma2 = 0, 2
x2 = np.linspace(mean2 - 6*sigma2, mean2 + 6*sigma2, 100)
 
mean3, sigma3 = 5, 1
x3 = np.linspace(mean3 - 6*sigma3, mean3 + 6*sigma3, 100)
 
y1 = normal_distribution(x1, mean1, sigma1)
y2 = normal_distribution(x2, mean2, sigma2)
y3 = normal_distribution(x3, mean3, sigma3)
 
plt.plot(x1, y1, 'r', label='m=0,sig=1')
plt.plot(x2, y2, 'g', label='m=0,sig=2')
plt.plot(x3, y3, 'b', label='m=1,sig=1')
plt.legend()
plt.grid()
plt.show()

#! 绘制一个3D高斯曲线 便于理解
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(x, y)
w0 = 1
gaussian = np.exp(-((pow(x, 2) + pow(y, 2)) / pow(w0, 2)))

#! 便于理解： 二维面振幅分布图
plt.figure()
plt.imshow(gaussian)

#! 便于理解： 三维曲面振幅分布图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, gaussian, cmap='jet')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 中值滤波
# 相当于用中值代替
#! 把filter中的数字从小到大排序：24，75，78，104，113，121，154
#! 找到中间值 113，把中间值113当作平滑处理后的结果。
median = cv2.medianBlur(img, 5)  # 中值滤波
#! 指定filter大小 （5 * 5）

#! 很好用，处理后所有噪音点都不见了！！

cv2.imshow('median', median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 展示所有的
res = np.hstack((blur,aussian,median))
#! blur均值,aussian高斯,median中值
# res = np.vstack((blur,aussian,median))
#! 通过hstack（横向）或者vstack（纵向）把所有结果拼接在一起
#! 相当于把三维矩阵的数值拼接在一起

#print (res)
cv2.imshow('median vs average', res)
cv2.waitKey(0)
cv2.destroyAllWindows()