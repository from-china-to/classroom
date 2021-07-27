'''
版本更新内容：
1.0：   根据Otsu原理 自己写代码 实现Otsu算法
1.2：   将自己实现的Otsu算法 封装在myotsu方法中
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
cv_show(th1,'th1')
print("MyOtsu 得到的阈值：\t"+str(ret1))

# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

images = [img, 0, th1, img, 0, th2]
titles = ['Original', 'Histogram', 'my_Otsu:'+str(ret1),
         'Original', 'Histogram', "Otsu's"+str(ret2),]

for i in range(2):
    # 绘制原图
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制直方图plt.hist, ravel函数将数组降成一维
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])
    
    # 绘制阈值图
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()