
'''
2_1.2 可以划线区分前景后景

(2_1.2.1 开始尝试 标记点 区分前景后景(删除了其他功能))

'''
# python OpenCV GrabCut使用实例解析
# https://www.jb51.net/article/173972.htm

'''
  Python: 3.5.7
  opencv 3.x

  在图上用鼠标左键和右键标记前景和后景即可.
  如果需要重新标记图像,关闭程序重新运行.
'''

import cv2
import numpy as np
import time

img_src = './img/pikachu07.jpg'

drawing = False
mode = False

class GrabCut:
  def __init__(self, t_img):
    self.img = t_img
    self.img_raw = img.copy()
    self.img_width = img.shape[0]
    self.img_height = img.shape[1]
    self.scale_size = 640 * self.img_width // self.img_height
    if self.img_width > 640:
      self.img = cv2.resize(self.img, (640, self.scale_size), interpolation=cv2.INTER_AREA)
    self.img_show = self.img.copy()
    self.img_gc = self.img.copy()
    self.img_gc = cv2.GaussianBlur(self.img_gc, (3, 3), 0)
    self.lb_up = False
    self.rb_up = False
    self.lb_down = False
    self.rb_down = False
    self.mask = np.full(self.img.shape[:2], 2, dtype=np.uint8) #? shape[:2]  [:2]：前两个[0]~[1]
    '''
    numpy.full(shape, fill_value, dtype=None, order='C')
    返回给定形状和类型的新数组，并用fill_value填充

    shape： ： int 或 sequence of ints  新阵列的形状，例如(2, 3)或者2。
    fill_value： ： scalar      填充值。
    dtype： ： data-type, 可选参数   数组所需的数据类型默认值，没有.    表示np.array(fill_value).dtype。
    order： ： {‘C’, ‘F’}, 可选参数    是否以C或Fortran-contiguous(行或列)顺序存储多维数据。

    返回值：	
    out： ： ndarray    具有给定形状，dtype和顺序的fill_value数组
    '''
    self.firt_choose = True


# 鼠标的回调函数
def mouse_event2(event, x, y, flags, param):
#?     MouseCallback onMouse的函数原型：
#! void on_Mouse(int event, int x, int y, int flags, void* param)
#! event是 CV_EVENT_*变量之一
#! x和y是鼠标指针在图像坐标系的坐标（不是窗口坐标系） 
#! flags是CV_EVENT_FLAG的组合， param是用户定义的传递到setMouseCallback函数调用的参数。
  global drawing, last_point, start_point
  # 左键按下：开始画图
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    last_point = (x, y)
    start_point = last_point
    param.lb_down = True
    print('mouse lb down')
  elif event == cv2.EVENT_RBUTTONDOWN:
    drawing = True
    last_point = (x, y)
    start_point = last_point
    param.rb_down = True
    print('mouse rb down')
  # 鼠标移动，画图
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing:
      if param.lb_down:
        cv2.line(param.img_show, last_point, (x,y), (0, 0, 255), 2, -1)
        '''
        cv2.line()方法用于在任何图像上绘制一条线。
        用法： cv2.line(image, start_point, end_point, color, thickness)
        参数：
        image:它是要在其上绘制线条的图像。
        start_point：它是线的起始坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
        end_point：它是直线的终点坐标。坐标表示为两个值的元组，即(X坐标值ÿ坐标值)。
        color:它是要绘制的线条的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
        thickness:它是线的粗细像素。
        返回值：它返回一个图像。
        '''
        cv2.rectangle(param.mask, last_point, (x, y), 1, -1, 4)
        '''
        cv2.rectangle 这个函数的作用是在图像上绘制一个简单的矩形
        Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → None
        img – Image.
        pt1 和 pt2 参数分别代表矩形的左上角和右下角两个点，而且 x 坐标轴是水平方向的，y 坐标轴是垂直方向的
        color 参数一般用 RGB 值指定，表示矩形边框的颜色。RGB 对应的颜色可以使用 https://www.sioe.cn/yingyong/yanse-rgb-16/ 查看。
        thickness 参数表示矩形边框的厚度，如果为负值，如 CV_FILLED，则表示填充整个矩形
        lineType –指定 Bresenham 算法是 4 连通的还是 8 连通的，涉及到了计算机图形学的知识。如果指定为 CV_AA，则是使用高斯滤波器画反锯齿线.
        8 (or omitted) - 8-connected line.
        4 - 4-connected line.
        CV_AA - antialiased line.
        shift – shift 参数表示点坐标中的小数位数，但是我感觉这个参数是在将坐标右移 shift 位一样。shift 为 1 就相当于坐标全部除以 2 1 2^12 1，shift 为 2 就相当于坐标全部除以 2 2 2^22 2
        原文链接：https://blog.csdn.net/sinat_41104353/article/details/85171185

        所谓四连通区域或四邻域，是指对应像素位置的上、下、左、右，是紧邻的位置。共4个方向，所以称之为四连通区域，又叫四邻域。
        所谓八连通区域或八邻域，是指对应位置的上、下、左、右、左上、右上、左下、右下，是紧邻的位置和斜向相邻的位置。共8个方向，所以称之为8连通区域或八邻域。
        https://blog.csdn.net/yewei11/article/details/50575593/
        '''
      else:
        cv2.line(param.img_show, last_point, (x, y), (255, 0, 0), 2, -1)
        cv2.rectangle(param.mask, last_point, (x, y), 0, -1, 4)
      last_point = (x, y)
  # 左键释放：结束画图
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False
    param.lb_up = True
    param.lb_down = False
    cv2.line(param.img_show, last_point, (x,y), (0, 0, 255), 2, -1)
    if param.firt_choose:
      param.firt_choose = False
    cv2.rectangle(param.mask, last_point, (x,y), 1, -1, 4)
    print('mouse lb up')
  elif event == cv2.EVENT_RBUTTONUP:
    drawing = False
    param.rb_up = True
    param.rb_down = False
    cv2.line(param.img_show, last_point, (x,y), (255, 0, 0), 2, -1)
    if param.firt_choose:
      param.firt_choose = False
      param.mask = np.full(param.img.shape[:2], 3, dtype=np.uint8)
    cv2.rectangle(param.mask, last_point, (x,y), 0, -1, 4)
    print('mouse rb up')

if __name__ == '__main__':
  img = cv2.imread(img_src)
  if img is None:
    print('error: 图像为空')
  g_img = GrabCut(img)

  cv2.namedWindow('image')
  # 定义鼠标的回调函数
  print('\n\n\t\t321:')
  cv2.setMouseCallback('image', mouse_event2, g_img)    #? 前面定义的鼠标事件都在这里setMouseCallback
  print(mouse_event2)
  '''
   void setMousecallback(const string& winname, MouseCallback onMouse, void* userdata=0)
   winname:窗口的名字
   onMouse:鼠标响应函数，回调函数。指定窗口里每次鼠标时间发生的时候，被调用的函数指针。 
   这个函数的原型应该为void on_Mouse(int event, int x, int y, int flags, void* param);
   userdate：传给回调函数的参数 

   void on_Mouse(int event, int x, int y, int flags, void* param);
   event是 CV_EVENT_*变量之一
   x和y是鼠标指针在图像坐标系的坐标（不是窗口坐标系） 
   flags是CV_EVENT_FLAG的组合， param是用户定义的传递到setMouseCallback函数调用的参数。
   
   #?附常用的event：
   #!defineCV_EVENT_MOUSEMOVE
   #!defineCV_EVENT_LBUTTONDOWN 
   #!defineCV_EVENT_RBUTTONDOWN   
   #!defineCV_EVENT_LBUTTONUP    
   #!defineCV_EVENT_RBUTTONUP   
   和标志位flags有关的：
   #defineCV_EVENT_FLAG_LBUTTON 
   注意： flags & CV_EVENT_FLAG_LBUTTON 的意思是 提取flags的CV_EVENT_FLAG_LBUTTON 标志位，!（）的意思是 标志位无效
   https://blog.csdn.net/qq_29540745/article/details/52562101

  fps设置要小，否则后面的帧在屏幕上覆盖了前面加了字符的帧图像，无法看清字符；
  setMouseCallback 必须在每次读到新帧后设置，否则回调函数中收到的参数param就不是指向当前帧，而是调用回调函数时frame变量对应帧，有可能frame还没定义或者定义的初始值如None，这样后面的鼠标点击无法触发对当前帧的操作；
  setMouseCallback调用时必须在窗口已经通过cv2.namedWindow或cv2.imshow定义了窗口名字之后，否则窗口没有定义回调函数设置没有作用；
  回调函数对当前帧添加了字符之后，需要再次调用cv2.imshow该帧才会刷新显示。
  https://blog.csdn.net/LaoYuanPython/article/details/108176864

    #?  event 具体说明如下：
    EVENT_MOUSEMOVE 0 //滑动
    EVENT_LBUTTONDOWN 1 //左键点击
    EVENT_RBUTTONDOWN 2 //右键点击
    EVENT_MBUTTONDOWN 3 //中键点击
    EVENT_LBUTTONUP 4 //左键放开
    EVENT_RBUTTONUP 5 //右键放开
    EVENT_MBUTTONUP 6 //中键放开
    EVENT_LBUTTONDBLCLK 7 //左键双击
    EVENT_RBUTTONDBLCLK 8 //右键双击
    EVENT_MBUTTONDBLCLK 9 //中键双击
    #?  flags 具体说明如下：
    EVENT_FLAG_LBUTTON 1 //左键拖曳
    EVENT_FLAG_RBUTTON 2 //右键拖曳
    EVENT_FLAG_MBUTTON 4 //中键拖曳
    EVENT_FLAG_CTRLKEY 8 //(8~15)按 Ctrl 不放
    EVENT_FLAG_SHIFTKEY 16 //(16~31)按 Shift 不放
    EVENT_FLAG_ALTKEY 32 //(32~39)按 Alt 不放
    #!  https://copyfuture.com/blogs-details/20210107234054522n
  '''
  while (True):
    cv2.imshow('image', g_img.img_show)
    if g_img.lb_up or g_img.rb_up:
      g_img.lb_up = False
      g_img.rb_up = False
      start = time.process_time()
      bgdModel = np.zeros((1, 65), np.float64)  #? bg模型的临时数组  13 * iterCount
      fgdModel = np.zeros((1, 65), np.float64)  #? fg模型的临时数组  13 * iterCount

      rect = (1, 1, g_img.img.shape[1], g_img.img.shape[0])
      print(g_img.mask)
      mask = g_img.mask
      g_img.img_gc = g_img.img.copy()
      cv2.grabCut(g_img.img_gc, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)   #? 真正的接口在这里
      #!   g_img.img_gc :g_img.img_gc = g_img.img.copy() 复制图像
      #!   mask :mask = g_img.mask 图像蒙版 
      #!   rect :rect = (1, 1, g_img.img.shape[1], g_img.img.shape[0]) #? 矩形，(x,y,w,h) #? ( y_min : y_min + h , x_min  : x_min + w)
      #!   bgdModel :bgdModel = np.zeros((1, 65), np.float64)  #? bg模型的临时数组  13 * iterCount
      #!   fgdModel :fgdModel = np.zeros((1, 65), np.float64)  #? fg模型的临时数组  13 * iterCount
      #!   5    :iterCount:指定迭代次数5
      #!   cv2.GC_INIT_WITH_MASK    :cv::GC_INIT_WITH_MASK//用掩码初始化grabCut
      elapsed = (time.process_time() - start)
      mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') # 0和2做背景
      g_img.img_gc_f = g_img.img_gc * mask2[:, :, np.newaxis] # 使用蒙板来获取前景区域
      cv2.imshow('foreground', g_img.img_gc_f)

      #! 仿写上面代码 输出背景：
      mask2_b = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8') # 0和2做背景
      g_img.img_gc_b = g_img.img_gc * mask2_b[:, :, np.newaxis] # 使用蒙板来获取前景区域
      cv2.imshow('background', g_img.img_gc_b)

    #   #? 工具接口 从这里保存图片进行下一步操作
    #   if cv2.waitKey(0) == 27:         # 按下esc时，退出
    #       cv2.destroyAllWindows()
    #   elif cv2.waitKey(0) == ord('s'): # 按下s键时保存并退出
    #       cv2.imwrite('./img/far_cut.png',g_img.img_gc)
    #       print("Saved Successfully !")
    #       cv2.destroyAllWindows()

      print("Time used:", elapsed)

    # 按下ESC键退出
    if cv2.waitKey(20) == 27:
      break

    #? 工具接口 从这里保存图片进行下一步操作
    elif cv2.waitKey(20) == ord('s'): # 按下s键时保存并退出
    
        cv2.imwrite('./img/pikachu07_cutforeground.jpg',g_img.img_gc_f) #保存前景
        cv2.imwrite('./img/pikachu07_cutbackground.jpg',g_img.img_gc_b) #保存背景
        #! '历史'遗留‘bug’ 把行业通用的 黑底白图 改成 白底黑图
        img_xy = g_img.img_gc_f   #! 第一步 读数据
        gray = cv2.cvtColor(img_xy, cv2.COLOR_BGR2GRAY)    #! 第二步 把数据转换成灰度图
        # ret, img_xy = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # m,n =img_xy.shape    
        # for j in range(n):  #行
        #     for i in range(m):  #列
        #         if(img_xy[i][j]==0):
        #             img_xy[i][j] = 255
        #         else:
        #             img_xy[i][j] =0
        
        ret, img_xy = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  #! 上述废话 等价于这个代码

        cv2.imwrite('./img/pikachu07_cut2.jpg',img_xy)
        print(g_img.img_gc_f)
        print("Saved Successfully !")