import numpy as np
import matplotlib.pyplot as plt
from skimage import io

################################### model #####################################
mean_filter3x3 = np.array([[1,1,1], [1,1,1], [1,1,1]]) / 9
mean_filter5x5 = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],
                           [1,1,1,1,1],[1,1,1,1,1]]) / 25
Gaussian3x3 = np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16
Gaussian5x5 = np.array([[1,4,7,4,1], [4,16,26,16,4], [7,26,41,26,7],
                        [4,16,26,16,4], [1,4,7,4,1]]) / 273
Laplace = np.array([[0,1,0],[1,-4,1],[0,1,0]])
Sobelx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
Sobely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


def Filter3x3(square, mode):
    '''
    Input:  square, a 3x3 numpy array
            mode,   can be mean, maximum, minimum, 
                            median, Gaussian, Laplace, Sobelx, Sobely
    Output: result value
    '''
    
    if mode == 'mean':
        return np.sum(square * mean_filter3x3)
    elif mode == 'maximum':
        return np.max(square)
    elif mode == 'minimum':
        return np.min(square)
    elif mode == 'median':
        return sorted(square.reshape(9))[4]
    elif mode == 'Gaussian':
        return np.sum(square * Gaussian3x3)
    elif mode == 'Laplace':
        return np.sum(square * Laplace)
    elif mode == 'Sobelx':
        return np.sum(square * Sobelx)
    elif mode == 'Sobely':
        return np.sum(square * Sobely)
    
    
def Filter5x5(square, mode):
    '''
    Input:  square, a 5x5 numpy array
            mode,   can be mean, maximum, minimum, median, Gaussian
    Output: result value
    '''
    
    if mode == 'mean':
        return np.sum(square * mean_filter5x5)
    elif mode == 'maximum':
        return np.max(square)
    elif mode == 'minimum':
        return np.min(square)
    elif mode == 'median':
        return sorted(square.reshape(25))[12]
    elif mode == 'Gaussian':
        return np.sum(square * Gaussian5x5)
    
    
def Spatial_filtering(img, filter_size, mode):
    '''
    Input:  img,         an numpy image
            filter_size, 3 or 5
            mode,        can be mean, maximum, minimum, median, Gaussian
    Output: image after filtering
    '''
    
    m, n = img.shape
    if filter_size == 3:
        padding = np.zeros((m+2,n+2))
        padding[1:m+1, 1:n+1] = img
        x = [1,m+1]
        y = [1,n+1]
    elif filter_size == 5:
        padding = np.zeros((m+4,n+4))
        padding[2:m+2, 2:n+2] = img
        x = [2,m+2]
        y = [2,n+2]
    
    new_img = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if filter_size == 3:
                x = i + 1
                y = j + 1
            else:
                x = i + 2
                y = j + 2
            square = padding[x-int(filter_size/2):x+int(filter_size/2)+1,
                   y-int(filter_size/2):y+int(filter_size/2)+1]
            if filter_size == 3:
                new_img[i][j] = Filter3x3(square, mode)
            else:
                new_img[i][j] = Filter5x5(square, mode)
    return new_img

#################################### test #####################################
image_path = 'D:/图片/'
image_name = '1.jpg'
img = io.imread(image_path + image_name, as_gray = True) # read as gray image

print('The original image')
plt.figure(figsize = (12,20))
plt.imshow(img ,cmap="gray")
plt.axis("off")
plt.show()

# set your parameter,  
# filter size: 3 or 5 
# mode: mean, maximum, minimum, median, Gaussian, Laplace, Sobelx, Sobely
filter_size = 3
mode = 'median'

new_img = Spatial_filtering(img, filter_size, mode)
print('Filtered image in {} mode, the filter size is {}.'.format(mode, filter_size))
plt.figure(figsize = (12,20))
plt.imshow(new_img ,cmap="gray")
plt.axis("off")
plt.show()





