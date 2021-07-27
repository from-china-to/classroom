# https://zhuanlan.zhihu.com/p/144807771
'''
OpenCV Findcontours( ) 函数原理出自于该论文的算法：
Topological Structural Analysis of Digitized Binary Images by Border Following
文章传送门：https://link.zhihu.com/?target=http%3A//pdf-s3.xuebalib.com%3A1262/1ftg5E69C3uX.pdf
'''
"""
Created on Wed May 27 15:01:45 2020

@author: 73766
"""    
import matplotlib.pyplot as plt
import numpy as np
#import cv2

#class Contour:
#    def __init__(self,parent,cur_num,contour_type):
#        self.parent = parent
#        self.contour_num = cur_num
#        self.contour_type = contour_type #Hole/Outer

class FindContours:
    def __init__(self):
        self.grid = np.array([[1,1,1,1,1,1,1,0,0],
                               [1,0,0,1,0,0,1,0,1],
                               [1,0,0,1,0,0,1,0,0],
                               [1,1,1,1,1,1,1,0,0]])
        self.reset()
        
    
    def reset(self):
        self.grid = np.pad(self.grid, ((1, 1), (1, 1)), 'constant', constant_values=0)
        self.LNBD = 1
        self.NBD = 1
        self.Disp_with_number = True
        self.MAX_BODER_NUMBER = self.grid.shape[0]*self.grid.shape[1]
        self.contours_dict = {}
        self.contours_dict[1] = self.Contour(-1,"Hole")
        
    def Contour(self,parent,contour_type,start_point = [-1,-1]):
        contour = {"parent":parent,
                   "contour_type":contour_type,
                   "son":[],
                   "start_point":start_point}#Hole/Outer
        return contour
    
    def load_map_from_array(self,grid):
        self.grid  = grid.copy().astype("int32")
        self.reset()
        
    def trans_number_to_char(self,num):
        if self.Disp_with_number:
            return str(num)
        if num >1:
            return chr(63 + num)
        if num <0:
            return chr(95 - num)
        else:
            return str(num)
    
    '''display gridd '''
    def disp_grid(self):
        for i in range(self.grid.shape[0]):
            num = '\033[0;37m' + '['
            print(num,end = ' ')
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == 0:
                    num = '\033[0;37m' + self.trans_number_to_char(self.grid[i][j]) 
                    print(num,end = ' ')
                else:
                    num = '\033[1;31m' + self.trans_number_to_char(self.grid[i][j]) 
                    print(num,end = ' ')
            num = '\033[0;37m' + ']'
            print(num)
        print("\033[0;37m")

        
    def find_neighbor(self,center,start,clock_wise = 1):
        weight = -1
        if clock_wise == 1:
            weight = 1
        #direction = np.array([[1,0],[0,-1],[0,-1],[-1,0],[-1,0],[0,1],[0,1]])
        neighbors = np.array([[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0]])
        indexs = np.array([[0,1,2],
                          [7,9,3],
                          [6,5,4]])
        #print(center,start)
        start_ind = indexs[start[0] - center[0]+1][start[1] - center[1]+1]
        # print(start_ind)
        for i in range(1,len(neighbors)+1): 
            cur_ind = (start_ind + i*weight+8)%8
            #print(cur_ind)
            x = neighbors[cur_ind][0] + center[0] - 1
            y = neighbors[cur_ind][1] + center[1] - 1
            # grid[x][y] = a
            # a+=1
            if self.grid[x][y] != 0:
                return [x,y]
        return [-1,-1]
    
    def board_follow(self,center_p,start_p,mode):
        ij = center_p
        ij2 = start_p
        ij1 = self.find_neighbor(ij,ij2,1)
        x = ij1[0]
        y = ij1[1]
        if ij1 == [-1,-1]:
                self.grid[ij[0]][ij[1]]  = -self.NBD
                return
        ij2 = ij1
        ij3 = ij
        for k in range(self.MAX_BODER_NUMBER):
            #step 3.3
            ij4 = self.find_neighbor(ij3,ij2,0)
            x = ij3[0]
            y = ij3[1]
            if ij4[0] - ij2[0] <=0:
                weight = -1
            else:
                weight = 1
            if self.grid[x][y] < 0:
                self.grid[x][y] = self.grid[x][y]
                
            elif self.grid[x][y-1] == 0 and self.grid[x][y+1] ==0:
                self.grid[x][y] = self.NBD*weight
  
            elif self.grid[x][y+1]== 0:
                self.grid[x][y] = -self.NBD
                
            elif self.grid[x][y]== 1 and self.grid[x][y+1] != 0:
                self.grid[x][y] = self.NBD
                
            else:
                self.grid[x][y] = self.grid[x][y]
                
            if ij4 == ij and ij3 ==ij1:
                return 
            ij2 = ij3
            ij3 = ij4
    
    def raster_scan(self):
        #self.disp_grid()
        for i in range(self.grid.shape[0]):
            self.LNBD = 1
            for j in range(self.grid.shape[1]):
                if abs(self.grid[i][j]) > 1:
                        self.LNBD = abs(self.grid[i][j])
                if self.grid[i][j] >= 1:
                    if self.grid[i][j] == 1 and self.grid[i][j-1] == 0:
                        self.NBD += 1
                        self.board_follow([i,j],[i,j-1],1)
                        border_type = "Outer"
                       
                        
                    elif self.grid[i][j] > 1 and self.grid[i][j+1] == 0:
                        border_type = "Hole"
                        #print(i,j)
                        self.NBD += 1
                        self.board_follow([i,j],[i,j+1],1)
                        #self.contours_dict[self.NBD] = self.Contour(self.LNBD,border_type)
                        #self.disp_grid()
                    else:
                        continue
                    parent = self.LNBD
                    if self.contours_dict[self.LNBD]["contour_type"] == border_type:
                        parent = self.contours_dict[self.LNBD]["parent"]
                    self.contours_dict[self.NBD] = self.Contour(parent,border_type,[i-1,j-1])
                    self.contours_dict[parent]["son"].append(self.NBD)

                    #print("NBD",self.NBD,"LNBD",self.LNBD)
                    
        self.grid = self.grid[1:-1,1:-1]


def main(): 
    fc = FindContours()       
    fc.raster_scan()
    fc.disp_grid()
    print(fc.contours_dict)
    
    grid1 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0,0,0,0,0,0],
                      [0,0,1,1,1,1,1,1,1,0,0,0,0],
                      [0,0,1,0,0,1,0,0,0,1,1,0,0],
                      [0,0,1,0,0,1,0,0,1,0,0,0,0],
                      [0,0,1,0,0,1,0,0,1,0,0,0,0],
                      [0,0,1,1,1,1,1,1,1,0,0,0,0],
                      [0,0,0,1,0,0,1,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0],])
    fc.load_map_from_array(grid1)
    fc.raster_scan()
    fc.disp_grid()
    print(fc.contours_dict)
    
    
#    
#    img1 = cv2.imread("D:\\datas\\luoxuan1.png")
#    img = np.mean(np.float32(img1), axis=2)
#    img[img<130] = 0
#    img[img>0] = 1
#    img = 1-img
#    
#    fc.load_map_from_array(img)
#    fc.raster_scan()
#    ret =abs(fc.grid) 
#    ret[ret<2] = 0
#    ret[ret>0] = 1
#    plt.figure()
#    plt.imshow(img,"gray") # 显示图片
#    plt.axis('off') # 不显示坐标轴
#    plt.show()
#    plt.figure()
#    plt.imshow(ret,"gray") # 显示图片
#    plt.axis('off') # 不显示坐标轴
#    plt.show()

if __name__ == "__main__":
    main()