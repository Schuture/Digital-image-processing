from skimage import io, color, img_as_ubyte
import numpy as np
from matplotlib import pyplot as plt

original = io.imread('D:/图片/equalization1.jpg')
original_gray = img_as_ubyte(color.rgb2gray(original))
H, W = original_gray.shape

hist = np.array([0] * 256, dtype = np.float32)
for i in range(H): #计算灰度直方图
    for j in range(W):
        hist[original_gray[i,j]] += 1
hist /= (H * W) # 归一化，变像素数量为像素比例

accumulated_hist = np.zeros(256, dtype = np.float32) # 计算累积灰度直方图      
accumulated_hist[0] = hist[0] 
for i in range(1,256):
    accumulated_hist[i] += hist[i] + accumulated_hist[i-1]
    
new_hist = np.zeros(256, dtype = np.float32) # 均衡后的灰度直方图
map_func = np.zeros(256, dtype = np.uint8) # 记录原像素值是如何映射到新像素的
for i in range(256): 
    map_func[i] = int(round(accumulated_hist[i] * 255))
    new_hist[map_func[i]] += hist[i] # 投影后的像素值增加投影前像素占全像素比例

new = np.array([[0] * W] * H)
for i in range(H): # 创建新图
    for j in range(W):
        new[i, j] = map_func[original_gray[i, j]]

print('原来的灰度直方图')
plt.figure(figsize=(20, 16))
plt.bar(np.arange(0,256),hist)
plt.xlabel('灰度')
plt.ylabel('频数')
plt.show()

print('新的灰度直方图')
plt.figure(figsize=(20, 16))
plt.bar(np.arange(0,256), new_hist)
plt.xlabel('灰度')
plt.ylabel('频数')
plt.show()

print('原图')
plt.figure(figsize=(20, 20))
plt.imshow(original, cmap = 'gray')
plt.show()

print('灰度图')
plt.figure(figsize=(20, 20))
plt.imshow(original_gray, cmap = 'gray')
plt.show()

print('均衡后的灰度图')
plt.figure(figsize=(20, 20))
plt.imshow(new, cmap = 'gray')
plt.show()