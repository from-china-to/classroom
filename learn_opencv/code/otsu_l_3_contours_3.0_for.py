
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageStat
# from __future__ import division
import __future__

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def my_contours(img):
    imgx=img.copy()
    imgy=img.copy()
    
    m,n =imgx.shape
    print(imgx.shape)
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
            if(img_xy[i-1][j-1]==254):
                img_xy[i-1][j-1] = 255
            else:
                img_xy[i-1][j-1] =0
                c_x.append(i-1)
                c_y.append(j-1)
    # print(img)
    # cv_show(img_xy,'img_xy')
    # print(c_x)  #? 怎么会有-1？？？
    # print(c_y)

    return img_xy,imgx,imgy,c_x,c_y 


#? Q4 自己写计算物理轮廓的算法 来提取分割区域的轮廓
img = cv2.imread('./img/whale_otsued.png',0)
img_xy,img_x,img_y,c_x,c_y=my_contours(img)

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
plt.imshow(cv2.imread('./img/whale_otsued.png',0),cmap="gray")
plt.axis('off')
plt.show()