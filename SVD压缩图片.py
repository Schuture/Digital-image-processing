import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 24))
from PIL import Image

######################设置参数#####################
r = 100 #要取前r个主成分，最多为图像短边的长度

#####################读取图片######################
img = Image.open('D:/图片/1.JPG')
I_gray = np.array(img.convert('L')) #转化为灰度图
m,n = np.shape(I_gray)[0],np.shape(I_gray)[1] #图像的大小

###################取前r个主成分计算新图####################
U,Sigma,V = np.linalg.svd(I_gray)
for i in range(r):
    Ui = U[:,i].reshape(m,1)
    Vi = V[i].reshape(1,n)
    if i==0:
        img_new = Sigma[i]*Ui.dot(Vi)
    else:
        img_new += Sigma[i]*Ui.dot(Vi)

#######################显示图片######################
plt.subplot(121)
plt.title('origin')
plt.imshow(img)

plt.subplot(122)
plt.title('compressed')
plt.imshow(img_new)
plt.show()