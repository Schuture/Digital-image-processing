import numpy as np
import copy
import matplotlib.pyplot as plt
import time

time_start=time.time()

plt.figure(figsize=(20, 24))

pic = plt.imread('D:/学习/课程/大数据/数值算法/2019课件/b1.JPG') #读取后就是array格式
plt.subplot(121)
plt.imshow(pic)
m,n = np.shape(pic)[0],np.shape(pic)[1]
data = pic.reshape(-1, 3) #将二维图展开成一维，通道数仍然为3，变成[m*n,3]

def kmeans_wave(steps, k, data):  #step为迭代次数，k为聚类数目，data为输入数据
    data_new = copy.deepcopy(data)
    data_new = np.hstack((data_new, np.ones([m*n,1]))) #扩展一个维度用来存放标签,[m*n,4]
    center_point = np.random.choice(m*n, k, replace=False) #随机选择初始点,长度为k的array
    center = data_new[center_point,:3] #[1,3]
    distance = [[] for i in range(k)] #距离度量,一个distance存着图中所有点到中心点的距离，[2,m*n]
    for i in range(steps):
         for j in range(k): # 计算到k个类的距离
             distance[j] = np.sqrt(np.sum(np.square(data_new[:,:3] - np.array(center[j])), axis=1)) #更新距离
         data_new[:,3] = np.argmin(np.array(distance), axis=0) #将最小距离的类别标签作为当前像素的类别
         for l in range(k): # 更新k个类的中心
             center[l] = np.mean(data_new[data_new[:,3]==l][:,:3], axis=0) #计算这一类聚类中心

    return data_new


data_new = kmeans_wave(100,2,data) #执行算法找到聚类中心

pic_new = data_new[:,3].reshape(m,n) #把标签值当作像素值
plt.subplot(122)
plt.imshow(pic_new)
time_end=time.time()
print('totally cost',time_end-time_start)