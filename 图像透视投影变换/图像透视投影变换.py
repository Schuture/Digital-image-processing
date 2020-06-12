# 功能：将图像中某一四边形变换为矩阵

import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_affine(img, bbox):
    # 源图像中四边形坐标点
    vertex = [[bbox[0],bbox[1]], [bbox[2],bbox[3]],
              [bbox[6],bbox[7]], [bbox[4],bbox[5]]] # 左上，左下，右上，右下
    w = max(vertex[1][0]-vertex[0][0], vertex[2][1]-vertex[0][1]) # 四边形最长边
    h = min(vertex[1][0]-vertex[0][0], vertex[2][1]-vertex[0][1]) # 四边形最短边
    
    point1 = np.array(vertex, dtype="float32")
    
    #转换后得到矩形的坐标点
    point2 = np.array([[0,0],[w,0],[0,h],[w,h]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(point1, point2)
    out_img = cv2.warpPerspective(img, M, (w,h))
    out_img = out_img[:,:,[2,1,0]]

    return out_img

if __name__ == '__main__':
    img = cv2.imread('img_9.jpg')
    gt = 'gt_img_9.txt'
    with open(gt) as f:
        while True:
            line = f.readline()
            if not line:
                break
            coordinates = line.split(',')[:8]
            label = line.split(',')[8]
            coordinates = [int(coor) for coor in coordinates]
            plt.axis('off') 
            plt.imshow(perspective_affine(img, coordinates))
            plt.show()
            print(label)
            
    
    
