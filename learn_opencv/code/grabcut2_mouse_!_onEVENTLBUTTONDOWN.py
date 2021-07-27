# python鼠标点击图片，获取点击点的像素坐标
# https://blog.csdn.net/ctgu361663454/article/details/102477279

import cv2
import numpy as np
#图片路径
img = cv2.imread('./img/pikachu03.jpg')
m_x = []
m_y = []
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        m_x.append(x)
        m_y.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        '''
        cv2.circle()方法用于在任何图像上绘制圆。
        用法： cv2.circle(image, center_coordinates, radius, color, thickness)
        参数：
        image:它是要在其上绘制圆的图像。
        center_coordinates：它是圆的中心坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
        radius:它是圆的半径。
        color:它是要绘制的圆的边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
        thickness:它是圆边界线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。
        返回值：它返回一个图像。
        '''
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
                    #? 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
        cv2.imshow("image", img)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
print(m_x[0],m_y[0])