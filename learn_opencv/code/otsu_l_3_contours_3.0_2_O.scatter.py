bicycles = [] 
print(bicycles) 
bicycles.append('ducati') 
print(bicycles) 

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2#x的平方

plt.figure()
plt.plot(x,y1) #画线
plt.scatter(x,y2) #画点

plt.figure(num=333,figsize=(8,5))#图333
plt.plot(x,y2)

plt.show()