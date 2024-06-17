import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt

def get_gaussian_blur_img(img, ksize=5, sigma=1.0):
    G = cv2.getGaussianKernel(ksize=ksize, sigma=sigma) # Guassian filter, ksize*1, value distribute according to guassian distribution
    G = np.outer(G,np.transpose(G)) # ksize * ksize filter
    return signal.convolve2d(img, G, boundary='symm', mode='same')

def part2_1(): # sharpen photo

    # step1: get blurred photo
    # step2: get edge detail (high pass) by (original photo - blurred photo)
    # step3: multiply edge detail by a cofficient -> add it back to orginal photo
    
    def unsharp_mask(img):
        r_img, g_img, b_img = cv2.split(img)
        blur_r_img = get_gaussian_blur_img(r_img, 3)
        blur_g_img = get_gaussian_blur_img(g_img, 3)
        blur_b_img = get_gaussian_blur_img(b_img)
        high_pass_r = r_img - blur_r_img
        high_pass_g = g_img - blur_g_img
        high_pass_b = b_img - blur_b_img
        mask = np.stack((high_pass_r, high_pass_g, high_pass_b), axis = 2)
        return mask

    def sharnpen(img, alpha): # alpha: sharpen cofficient
        print(np.clip(img + unsharp_mask(img) * alpha, a_min=0, a_max=255))
        return np.clip(img + unsharp_mask(img) * alpha, a_min=0, a_max=255) # clip: limit value in a_min, a_max    

    img = cv2.imread('intake/taj.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sharp_img = sharnpen(img, 0.8).astype(int)
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img)
    axes[1].imshow(sharp_img)
    plt.show()
    cv2.imwrite('output/taj_sharpen.jpg', sharp_img)

if __name__ == '__main__':

    part2_1()