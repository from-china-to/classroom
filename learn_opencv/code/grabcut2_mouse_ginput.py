'''
图像标记点的坐标输出功能
'''

from PIL import Image
from pylab import *

im = array(Image.open('./img/pikachu03.jpg'))
imshow(im)
print ('Please click 3 points')
x =ginput(3)
print ('you clicked:',x)
show()