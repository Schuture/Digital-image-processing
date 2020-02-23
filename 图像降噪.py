import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def Noise(img, mode):
    '''
    Add noise to an image
    
    Input:
        img: ndarray image
        mode: salt, Gaussian or uniform
    Output:
        image with noise
    '''
    M, N = img.shape
    if mode == 'Gaussian':
        random_num = np.random.randn(M,N) * 0.05
        new_img = img + random_num
        new_img[np.where(new_img<0)] = 0
        new_img[np.where(new_img>1)] = 1
        return new_img
    elif mode == 'uniform':
        random_num = np.random.uniform(-1, 1, (M,N)) * 0.1
        new_img = img + random_num
        new_img[np.where(new_img<0)] = 0
        new_img[np.where(new_img>1)] = 1
        return new_img
        
    new_img = img.copy()
    for i in range(M):
        for j in range(N):
            random_num = np.random.uniform(0, 1, 1)
            if random_num < 0.025:
                new_img[i,j] = 0.0
            elif random_num > 0.975:
                new_img[i,j] = 1.0
    return new_img
            

mean_filter3x3 = np.array([[1,1,1], [1,1,1], [1,1,1]]) / 9
Gaussian3x3 = np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16

def Filter3x3(square, mode):
    '''
    Input:  square, a 3x3 numpy array
            mode,   can be mean, median, Gaussian
    Output: result value
    '''
    
    if mode == 'mean':
        return np.sum(square * mean_filter3x3)
    elif mode == 'median':
        return sorted(square.reshape(9))[4]
    elif mode == 'Gaussian':
        return np.sum(square * Gaussian3x3)


def Spatial_filtering(img, mode):
    '''
    Input:  img,         an numpy image
            mode,        can be mean, median, Gaussian
    Output: image after filtering
    '''
    
    m, n = img.shape
    padding = np.zeros((m+2,n+2))
    padding[1:m+1, 1:n+1] = img
    x = [1,m+1]
    y = [1,n+1]
    
    new_img = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            x = i + 1
            y = j + 1

            square = padding[x-1:x+2,y-1:y+2]
            new_img[i][j] = Filter3x3(square, mode)

    return new_img


def test():
    image_path = 'D:/图片/'
    image_name = 'lena.jpg'
    img = io.imread(image_path + image_name, as_gray = True) # read as gray image

    print('Three kinds of noise:')
    # three types of noise
    salt = Noise(img, 'salt')
    Gaussian = Noise(img, 'Gaussian')
    uniform = Noise(img, 'uniform')
    
    plt.figure(figsize = (20,20))
    plt.subplot(221)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.subplot(222)
    plt.title('Salt and pepper noise')
    plt.imshow(salt, cmap='gray')
    plt.subplot(223)
    plt.title('Gaussian noise')
    plt.imshow(Gaussian, cmap='gray')
    plt.subplot(224)
    plt.title('uniform noise')
    plt.imshow(uniform, cmap='gray')
    plt.show()
    
    print('\n\nThree kinds of denoising:')
    # three types of denoising
    mean = Spatial_filtering(salt, 'mean')
    median = Spatial_filtering(salt, 'median')
    Gaussian = Spatial_filtering(salt, 'Gaussian')
    
    plt.figure(figsize = (20,20))
    plt.subplot(221)
    plt.title('Image with salt and pepper noise')
    plt.imshow(salt, cmap='gray')
    plt.subplot(222)
    plt.title('Denoising with mean filtering')
    plt.imshow(mean, cmap='gray')
    plt.subplot(223)
    plt.title('Denoising with median filtering')
    plt.imshow(median, cmap='gray')
    plt.subplot(224)
    plt.title('Denoising with Gaussian filtering')
    plt.imshow(Gaussian, cmap='gray')


if __name__ == '__main__':
    test()











