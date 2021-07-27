
#? python统计RGB图片某像素的个数

from PIL import Image
import numpy as np
import cv2

img_L = np.array(Image.open('./img/whale.png').convert("L"))
img_RGB = np.array(Image.open('./img/whale.png').convert("RGB"))
print(img_L)
print(img_RGB)
# temp = {}
# for i in range(img_L.shape[0]):
#     for j in range(img_L.shape[1]):
#         if not temp.get(int(img_L[i][j])):
#             temp[int(img_L[i][j])] = list(img_RGB[i][j])
# print(temp)

#这里得到灰度像素值0对应(0,0,0),62对应(19,69,139)
color_0_0_0 = np.where(img_L == 0)[0].shape[0]
color_19_69_139 = np.where(img_L == 62)[0].shape[0]

pixel_sum = img_L.shape[0] * img_L.shape[1]

print("0_0_0 像素个数：{} 占比：%{}".format(color_0_0_0,color_0_0_0/pixel_sum*100))
print("19_69_139 像素个数：{} 占比：%{}".format(color_19_69_139,color_19_69_139/pixel_sum*100))
