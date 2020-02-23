import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# get fague image
def Filter(image, mode = 'low', d = 30, Gaussian = False):
    '''
    Input:  image, MxN numpy array
            d,     a number, the frequency filter threshold
    Output: the transformed image
    '''
    m, n = image.shape
    f = np.fft.fft2(image) # Fourier transform, space domain to freq domain
    fshift = np.fft.fftshift(f) # centerize the signal
    
    def make_transform_matrix(d):
        '''
        Input:  distance d
        Output: mask for (Gaussian) lowpass, high pass
        '''
        def cal_distance(x, y, pb):
            dis = np.sqrt((x - pb[0])**2 + (y - pb[1])**2)
            return dis
        
        x = np.array([[i for j in range(n)] for i in range(m)])
        y = np.array([[j for j in range(n)] for i in range(m)])
        center_point = tuple(map(lambda x:(x-1)/2, fshift.shape))
        mask = cal_distance(x, y, center_point) # the distance to center
        
        
        if Gaussian:
            if mode == 'low':
                mask = np.exp(-(mask**2)/(2*(d**2)))
            else:
                mask = 1 - np.exp(-(mask**2)/(2*(d**2)))
        else:
            if mode == 'low':
                mask = np.where(mask<d, 1, 0)
            else:
                mask = np.where(mask>d, 1, 0)

        return mask
    
    d_matrix = make_transform_matrix(d)
    # inverse transform
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img

################################## test ####################################

image_path = 'D:/图片/'
image_name = '1.jpg'
img = io.imread(image_path + image_name, as_gray = True) # read as gray image

print('The original image')
plt.figure(figsize = (6,10))
plt.imshow(img ,cmap="gray")
plt.axis("off")
plt.show()

# transform parameters
d = [60, 30, 10]
mode = 'high'
Gaussian = True

print('The {}pass transformed image with d = {}'.format(mode, d[0]))
plt.figure(figsize = (6,10))
plt.imshow(Filter(img, mode, d[0], Gaussian), cmap="gray")
plt.axis("off")
plt.show()

print('The {}pass transformed image with d = {}'.format(mode, d[1]))
plt.figure(figsize = (6,10))
plt.imshow(Filter(img, mode, d[1], Gaussian), cmap="gray")
plt.axis("off")
plt.show()

print('The {}pass transformed image with d = {}'.format(mode, d[2]))
plt.figure(figsize = (6,10))
plt.imshow(Filter(img, mode, d[2], Gaussian), cmap="gray")
plt.axis("off")
plt.show()



















