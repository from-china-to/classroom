'''
版本更新内容：
1.0：   根据Otsu原理 自己写代码 实现Otsu算法（写有大量注释）
1.2：   将自己实现的Otsu算法 封装在myotsu方法中,并且删除大量注释和多于代码
2.0:    对比开闭运算处理的结果，选取开运算消除噪声点
3.0：   用学过的方法实现提取分割区域的轮廓（写有大量注释）
3.2：   删除了大量注释，保留了最精简的代码
3.contours.1:   单独 写一个计算contours的算法
3.contours.2:   单独 写一个计算contours的算法,优化边界问题
3.contours2.0： 代码部分优化 精简+注释(写有大量注释 并包含试错代码)
3.contours2.9： 封装my_contours，并 精简注释删除大部分无用代码
3.contours3.0： 准备提取的轮廓彩色化---在my_contours中添加代码获取每个轮廓点的X【。。。】Y【。。。】坐标列表  range从0开始计数的；要重新调for循环
3.contours3.9： R化的轮廓 在原图上进行标注  【最终汇报版】【保存有完整的代码】
3.contours3.9： 删除一切无用代码和功能【最终汇使用版】

'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageStat
# from __future__ import division
import __future__

from numpy.lib import npyio

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
        '''
        where[0]包含符合条件的所有x坐标，where[1]是y坐标，【0】【1】元素个数一样
        shape永远是[0] ，只有一个值 保存在shape[0] 保存的是复合条件的点的个数
        '''
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

def my_contours(img):
    imgx=img.copy()
    imgy=img.copy()
    
    m,n =imgx.shape
    # n = int(3)
    # j=int(3)
    #! 遍历行/列，遇见的第一个0保留，后面连续的0全部变成255，直到遇见255再把前一个变成255的0还原成0.
    for i in range(m):  #行
        # print(str(i)+"...\t...\t...\t")      
        for j in range(n):  #列
            if(imgx[i][j]==0 and j!=n-1):      #? 开头少一条黑线。仅仅海洋地方少一条黑线
                # print("\t:"+str(j))
                # print("\t:"+str(img[i-1][j-1-1]))
                # print("\t:"+str(img[i-1][j-1]))
                # print("\t:"+str(img[i-1][j-1+1]))
                while(imgx[i][j+1] == 0 and j+1!=n-1):
                    imgx[i][j+1] = 255
                    # print("\t\t-"+str(j))
                    j+=1
                if(j!=n-1):             #! 解决了，一句话搞定~~解决了 末尾双重黑问题
                    imgx[i][j] = 0  #?这里出问题，最后双重黑
                if(j+1==n-1 and imgx[i][j+1] == 0):
                    imgx[i][j] = 255    #! 改了三天bug 把第一根线改回来了 并且把 最后根线处理好了（还是最初的‘笨’方法）
                    break
                    # print("\t\t-"+str(j))
                    # print("\t\t-"+str(img[i-1][j-1-1]))
                    # print("\t\t-"+str(img[i-1][j-1]))
                    # print("\t\t-"+str(img[i-1][j-1+1]))
            # print(j)
    # print(img)
    # cv_show(imgx,'img_x')
    # img_x=imgx
    #! ↑X方向完成了

    m,n =imgy.shape
    for j in range(n):  #行
        # print(str(j)+"...\t...\t...\t")      
        for i in range(m):  #列
            if(imgy[i][j]==0 and i!=m-1):     #! 天撸了 and i!=m-1 这个也要加上啊，倒是第二个白色 最后单独一个是黑色的情况呀！！！ 
                # print("\t:"+str(i))
                while(imgy[i+1][j] == 0 and i+1!=m-1):
                    imgy[i+1][j] = 255
                    # print("\t\t-"+str(i))
                    i+=1
                if(i!=m-1):            
                    imgy[i][j] = 0  
                if(i+1==m-1 and imgy[i+1][j] == 0):
                    imgy[i][j] = 255   
                    break
    # print(img)
    # cv_show(imgy,'img_y')
    # img_y=imgy
    #! ↑y方向完成了

    # img_xy = img_x+img_y    #! 两图合并
    img_xy = imgx+imgy
    # cv_show(img_xy,'done_ing')
    # print(img_xy)

    #! ↓合并后的图像处理一下像素值 并保存contours的坐标列表
    m,n =img_xy.shape
    c_x = []    # 保存contours的x坐标
    c_y = []    # 保存contours的y坐标
    for j in range(n):  #行
        for i in range(m):  #列
            if(img_xy[i][j]==254):
                img_xy[i][j] = 255
            else:
                img_xy[i][j] =0
                c_x.append(i)
                c_y.append(j)
    # print(img)
    # cv_show(img_xy,'img_xy')
    # print(c_x)
    # print(c_y)

    return img_xy,imgx,imgy,c_x,c_y 

#? Q1 打开一幅图片
img = cv2.imread('./img/lena_cut2.jpg', 0)

#? Q2 自己写一个otsu算法 做阈值分割
# 我的otsu 阈值法
ret1, th1 = myotsu(img)
print("MyOtsu 得到的阈值：\t"+str(ret1))
'''
ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.threshold (src, thresh, maxval, type)
cv2.threshold (源图片, 阈值, 填充色, 阈值类型)
'''

# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
'''
ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.threshold (src, thresh, maxval, type)
cv2.threshold (源图片, 阈值, 填充色, 阈值类型)
'''

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


#? Q3 用形态学开/闭运算处理消除噪声点
#! 开运算与闭运算
kernel = np.ones((5,5),np.uint8) 
o_img = th1.copy()# 开：先腐蚀，再膨胀
opening = cv2.morphologyEx(o_img, cv2.MORPH_OPEN, kernel)

c_img = th1.copy()# 闭：先膨胀，再腐蚀
closing = cv2.morphologyEx(c_img, cv2.MORPH_CLOSE, kernel)

plt.figure()
plt.subplot(2,2,1)		# 将画板分为2行两列，本幅图位于第一个位置
plt.imshow(img,cmap="gray")
plt.axis('off')
plt.subplot(2,2,2)		# 将画板分为2行两列，本幅图位于第二个位置
plt.imshow(th1,cmap="gray")
plt.axis('off')
plt.subplot(2,2,3)		# 将画板分为2行两列，本幅图位于第3个位置
plt.imshow(opening,cmap="gray")
plt.axis('off')
plt.subplot(2,2,4)		# 将画板分为2行两列，本幅图位于第3个位置
plt.imshow(closing,cmap="gray")
plt.axis('off')
plt.show()


#? Q4 自己写计算物理轮廓的算法 来提取分割区域的轮廓
# img = cv2.imread('./img/whale_otsued.png',0)
img_xy,img_x,img_y,c_x,c_y=my_contours(opening)

plt.figure()
plt.subplot(2,2,1)		
plt.imshow(img_x,cmap="gray")
plt.axis('off')
plt.subplot(2,2,2)		
plt.imshow(img_y,cmap="gray")
plt.axis('off')
plt.subplot(2,2,3)		
plt.imshow(img_xy,cmap="gray")
plt.axis('off')
plt.subplot(2,2,4)
# plt.imshow(cv2.imread('./img/whale_otsued.png',0),cmap="gray")
plt.imshow(th1,cmap="gray")
plt.axis('off')
plt.show()

#? Q5 提取到的轮廓 叠加在原图上进行显示

#? Q5A1 兜兜转转 最初的思路最简单最有效：
#! （用myfind的XY坐标列表，直接修改原图像素点）
img = img_xy.copy()
imgin1 = cv2.imread('./img/lena.jpg')[:,:,::-1] #? [:,:,::-1] 
#![:,:,::-1] 【第一个通道，第二个通道，第三个通道】第三个通道-1意思是RGB变成BGR

sx,sy= img.shape
l =len(c_x)

x = c_x    # 保存contours的x坐标
y = c_y    # 保存contours的y坐标
for i in range(l):
    imgin1[x[i],y[i],0]=255
    imgin1[x[i],y[i],1]=0
    imgin1[x[i],y[i],2]=0

plt.imshow(imgin1)
plt.axis('off')
plt.show()

