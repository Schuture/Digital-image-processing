import time
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 24))
from PIL import Image

time_start = time.time()

img = Image.open('D:/学习/课程/数值算法/2019课件/b3.JPG')
I_gray = np.array(img.convert('L')) #转化为灰度图

plt.subplot(121)
plt.imshow(img)
m,n = np.shape(I_gray)[0],np.shape(I_gray)[1] #图像的大小

graylevel = 256 #灰度级

hist = np.zeros(graylevel) #初始化灰度直方图

for i in range(m): #计算灰度直方图 
    for j in range(n):
        l = I_gray[i,j] #图像的灰度级l,从0到255
        hist[l] += 1 #灰度值为l的像素数量

hist = hist/(m*n) #直方图归一化,就是计算每一级灰度的占比

miuT=0 #定义总体均值
for l in range(graylevel):
    miuT = miuT + l*hist[l] #总体均值

max_var = 0 #最大方差
var = [0 for i in range(graylevel)]

for level in range(20,graylevel-20): #不考虑阈值太极端的情况
    threshold = level #设定阈值
    
    omega1 = 0 
    for l in range(threshold): #低于阈值算第一类,大于等于算第二类
        omega1 += hist[l] #第一类概率（用占比来估计）
    omega2 = 1-omega1 #第二类概率
    
    miu1=0 #第一类的平均灰度值
    miu2=0 #第二类的平均灰度值
    for l in range(graylevel):
        if l<threshold:
            miu1 += l*hist[l] #第一类灰度的累加值
        else:
            miu2 += l*hist[l] #第二类灰度的累加值
    miu1 /= omega1 #第一类的平均灰度值
    miu2 /= omega2 #第二类的平均灰度值
    
    var[level] = omega1*(miu1-miuT)**2 + omega2*(miu2-miuT)**2 #当前阈值的方差
    
final = np.argmax(var) #找到使方差最大的那个灰度级

img_new = np.zeros([m,n])
for i in range(m): #生成分割图像
    for j in range(n):
        if I_gray[i][j]>final: #大于所设定的均值则为目标
            img_new[i][j] = 1
        else:
            img_new[i][j] = 0
plt.subplot(122)
plt.imshow(img_new)

time_end=time.time()
print('totally cost',time_end-time_start)