'''
直方图与傅里叶变换

傅里叶变换
'''

#  图像处理
#! 直方图与傅里叶变换

#? 傅里叶变换
'''
我们生活在时间的世界中，早上7:00起来吃早饭，8:00去挤地铁，9:00开始上班。。。以时间为参照就是时域分析。
但是在频域中一切都是静止的！
https://zhuanlan.zhihu.com/p/19763358
'''

#? 傅里叶变换的作用
'''
高频：变化剧烈的灰度分量，例如边界
低频：变化缓慢的灰度分量，例如一片大海
'''

#? 滤波
'''
低通滤波器：只保留低频，会使得图像模糊      /高频没了
高通滤波器：只保留高频，会使得图像细节增强  /低频没了 边界锐化

opencv中主要就是cv2.dft()和cv2.idft()，输入图像需要先转换成np.float32 格式。
#！【cv2.dft()傅里叶变换 cv2.idft()逆变换，因为频率不好展示所以需要逆变换】
得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
#！【把低频区 从‘原点左上方’ 拉到 ‘原点中心’】
cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。
#！【转换成 0-255的一个值】
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/lena.jpg',0)
#! 灰度图

img_float32 = np.float32(img)
#! openCV官方要求：输入图像需要先转换成np.float32 格式。

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
#! 传入图像，执行 傅里叶变换
dft_shift = np.fft.fftshift(dft)
#? 得到 频谱图,np中也有一个fft直接变换就行，把低频区的值 从‘原点左上方’ 拉到 ‘原点中心’这个中间位置
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#! 对【，，0】【，，1】两个通道进行转换；但是转换后的值非常小；我们需要把它转换成0-255的一个值 直接使用代码里这个公式进行转换就行

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#! 得到的结果 图片中心比较亮，距离中心点越近频值越低，越往外发散 频值越高
plt.show()
#?  ↑ dft之后的结果

#? ↓低通高通后的结果 
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

img = cv2.imread('./img/lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
#! 计算一下shape大小 和 中心位置
#? 为什么要计算中心位置
#! 相当于把 中心位置当作‘（0，0）’，然后中心位置范围‘30’‘-30，30；30，30；-30，-30；30，-30’的范围内白；其他地方黑
#! 这么的就做了个 低通滤波器，黑色过滤掉了 白色都留下来了

# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
#! 长和宽 与 图像一致
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#! 相当于只有中心位置是1 其他位置都是0

#?  之前展示的图 从左边lena 到 右边 变换，是dft； 从右到左变换是idft
# IDFT
fshift = dft_shift*mask
#! 先和掩码mask结合一下：是1的都可以保留下来了；不是1的都过滤掉了
f_ishift = np.fft.ifftshift(fshift)
#! 此时只留下一个中间区域，
#? np.fft.ifftshift因为 之前把值从左上角拿到了中间位置，现在要从中间还回原来的位置（左上角）
img_back = cv2.idft(f_ishift)
#! idft还原，此时的结果 还不是个图像，是个双通道的结果
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#! 把双通道的 实部和虚部进行处理

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()        
#! 低通处理后的结果，帽子里面的东西 人身体里面的东西 脸里面的东西保留下来了，边界变得模糊了


img = cv2.imread('./img/lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
#! 高频和低频其实一回事儿，高频就是 中间那部分不要了（黑），外面那一圈那部分 留下来（白）

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()    
#! 高通滤波器的结果 只保留了边界信息


#? 为什么非要 转换到一个频率中做处理呢？
#? eg:检测图像中 哪些是低频那些是高频？如果在原始图像中做 会特别麻烦，在频率中作变换 会容易 会频次分明
#? 所以 如果我们对象处理的时候 需要速度 需要简单高效 的时候
#? 我们会先把图像映射到频率当中，在频率当中做处理 比 原始图像中处理容易得多。
#? 
#? 