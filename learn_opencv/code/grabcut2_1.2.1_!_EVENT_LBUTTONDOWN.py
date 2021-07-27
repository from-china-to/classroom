
'''
!!!汇报版，用鼠标标记点 切割

2_1.2   可以划线区分前景后景
2_1.2.1 开始尝试 标记点 区分前景后景(删除了其他功能)
'''


import cv2
import numpy as np
import time

img_src = './img/pikachu03.jpg'

drawing = False
mode = False

m_x = []#c 存放鼠标X
m_y = []#c 存放鼠标y

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
    self.mask = np.full(self.img.shape[:2], 2, dtype=np.uint8)   #?c   numpy.full(shape, fill_value, dtype=None, order='C')
    self.firt_choose = True


# 鼠标的回调函数
def mouse_event2(event, x, y, flags, param):
#?     MouseCallback onMouse的函数原型：
#! void on_Mouse(int event, int x, int y, int flags, void* param)
#! event是 CV_EVENT_*变量之一
#! x和y是鼠标指针在图像坐标系的坐标（不是窗口坐标系） 
#! flags是CV_EVENT_FLAG的组合， param是用户定义的传递到setMouseCallback函数调用的参数。
  global drawing, last_point, start_point
  # 左键按下：开始画图；
  #c 显示 鼠标所在坐标
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    last_point = (x, y)
    start_point = last_point
    param.lb_down = True
    print('mouse lb down')
    print('start_point'+str(start_point))

    # xy = "%d,%d" % (x, y)
    # m_x.append(x)
    # m_y.append(y)
    # # cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
    # cv2.circle(img,(x,y),5,(255,255,0),-1)
    # # cv2.circle(img,(x,y),10,(255,255,0),2)
    # # 则以此时双击的点为原点画一个半径为100px BGR为(255,255,0)粗细为3px的圆圈
    # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
    #                 1.0, (0, 0, 0), thickness=1)
    # cv2.imshow("text", img)

  elif event == cv2.EVENT_RBUTTONDOWN:
    drawing = True
    last_point = (x, y)
    start_point = last_point
    param.rb_down = True
    print('-mouse rb down')
    print('-start_point'+str(start_point))
  # 鼠标移动，画图
#   elif event == cv2.EVENT_MOUSEMOVE:
#     if drawing:
#       if param.lb_down:
#         cv2.line(param.img_show, last_point, (x,y), (0, 0, 255), 2, -1)
#         cv2.rectangle(param.mask, last_point, (x, y), 1, -1, 4)
#       else:
#         cv2.line(param.img_show, last_point, (x, y), (255, 0, 0), 2, -1)
#         cv2.rectangle(param.mask, last_point, (x, y), 0, -1, 4)
#       last_point = (x, y)
  # 左键释放：结束画图
#   elif event == cv2.EVENT_LBUTTONUP:
  elif event == cv2.EVENT_LBUTTONDBLCLK:
    drawing = False
    param.lb_up = True
    param.lb_down = False
    # cv2.line(param.img_show, last_point, (x,y), (0, 0, 255), 2, -1)
    if param.firt_choose:
      param.firt_choose = False
    cv2.rectangle(param.mask, last_point, (x,y), 1, -1, 4)
    #c cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    print('mouse lb up')
    print('last_point'+str(last_point))

#   elif event == cv2.EVENT_RBUTTONUP:
  elif event == cv2.EVENT_RBUTTONDBLCLK:
    drawing = False
    param.rb_up = True
    param.rb_down = False
    # cv2.line(param.img_show, last_point, (x,y), (255, 0, 0), 2, -1)
    if param.firt_choose:
      param.firt_choose = False
      param.mask = np.full(param.img.shape[:2], 3, dtype=np.uint8)   #?c   numpy.full(shape, fill_value, dtype=None, order='C')
    cv2.rectangle(param.mask, last_point, (x,y), 0, -1, 4)
    print('-mouse rb up')
    print('-last_point'+str(last_point))

if __name__ == '__main__':
  img = cv2.imread(img_src)
  if img is None:
    print('error: 图像为空')
  g_img = GrabCut(img)

  cv2.namedWindow('image')
  # 定义鼠标的回调函数
  print('\n\n\t\tmouse_event2:')
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
    EVENT_LBUTTONDOWN 1 //左键点击
    EVENT_RBUTTONDOWN 2 //右键点击
    EVENT_LBUTTONUP 4 //左键放开
    EVENT_RBUTTONUP 5 //右键放开
    EVENT_LBUTTONDBLCLK 7 //左键双击
    EVENT_RBUTTONDBLCLK 8 //右键双击
    #!  https://copyfuture.com/blogs-details/20210107234054522n
  '''
  while (True):
    cv2.imshow('image', g_img.img_show)
    if g_img.lb_up or g_img.rb_up:
      g_img.lb_up = False
      g_img.rb_up = False

      bgdModel = np.zeros((1, 65), np.float64)  #? bg模型的临时数组  13 * iterCount
      fgdModel = np.zeros((1, 65), np.float64)  #? fg模型的临时数组  13 * iterCount

      rect = (1, 1, g_img.img.shape[1], g_img.img.shape[0])
      print('\n\n\t\trect:')
      print(rect)
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

      mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') # 0和2做背景
      g_img.img_gc_f = g_img.img_gc * mask2[:, :, np.newaxis] # 使用蒙板来获取前景区域
      cv2.imshow('foreground', g_img.img_gc_f)

      #! 仿写上面代码 输出背景：
      mask2_b = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8') # 0和2做背景
      g_img.img_gc_b = g_img.img_gc * mask2_b[:, :, np.newaxis] # 使用蒙板来获取前景区域
      cv2.imshow('background', g_img.img_gc_b)

    # 按下ESC键退出
    if cv2.waitKey(20) == 27:
      break
