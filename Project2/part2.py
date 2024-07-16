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
    sharp_img = sharnpen(img, 1.2).astype(np.uint8)
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img)
    axes[1].imshow(sharp_img)
    print(sharp_img.shape)
    plt.show()
    output_img = cv2.cvtColor(sharp_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output/taj_sharpen.jpg', output_img)

def part2_2():

    img_for_low = cv2.imread('intake/DerekPicture.jpg', cv2.IMREAD_COLOR)
    img_for_high = cv2.imread('intake/nutmeg.jpg', cv2.IMREAD_COLOR)

    def get_high_pass_img(img, ksize=9, sigma=11.0):
        b_img, g_img, r_img = cv2.split(img)
        blur_r_img = get_gaussian_blur_img(r_img, ksize=ksize, sigma=sigma)
        blur_g_img = get_gaussian_blur_img(g_img, ksize=ksize, sigma=sigma)
        blur_b_img = get_gaussian_blur_img(b_img, ksize=ksize, sigma=sigma)
        high_pass_img = np.stack((b_img - blur_b_img, g_img - blur_g_img, r_img - blur_r_img), axis=2)
        return high_pass_img

    low_pass_img = (img_for_low - get_high_pass_img(img_for_low)).astype(np.uint8)
    print(low_pass_img)
    high_pass_img = get_high_pass_img(img_for_high).astype(np.uint8)

    target_size = (low_pass_img.shape[1], low_pass_img.shape[0])
    high_pass_img = cv2.resize(high_pass_img, target_size)

    hybrid_img = (low_pass_img + high_pass_img * 0.1).astype(np.uint8)

    cv2.imshow('windows', hybrid_img)
    cv2.waitKey(0)
    cv2.imwrite('output/hybrid.jpg', hybrid_img)    

def part2_3():
    def get_stack(img, n):
        gaussian_stk = []
        laplacian_stk = []
        last_img = img

        gaussian_stk.append(img)
        for i in range(n-1):
            blur_img = get_gaussian_blur_img(last_img)
            gaussian_stk.append(blur_img)
            laplacian_stk.append(last_img - blur_img)
            last_img = blur_img
        laplacian_stk.append(last_img)
        return gaussian_stk, laplacian_stk

    img = cv2.imread('intake/orange.jpeg', cv2.IMREAD_GRAYSCALE)
    N = 5
    gaussian_stk, laplacian_stk = get_stack(img, N)

    fig, axes = plt.subplots(4,3)
    for i in range(N-1):
        axes[i][0].imshow(gaussian_stk[i])
        axes[i][1].imshow(laplacian_stk[i])
        axes[i][2].imshow(gaussian_stk[i+1])
    plt.show()

def part2_4():
        
    def get_stack(img, n):
        gaussian_stk = []
        laplacian_stk = []
        last_img = img
        gaussian_stk.append(img)

        for i in range(n-1):
            blur_img = get_gaussian_blur_img(last_img, 20, 15)
            gaussian_stk.append(blur_img)
            laplacian_stk.append(last_img - blur_img)
            last_img = blur_img
        laplacian_stk.append(last_img)
        return gaussian_stk, laplacian_stk
    
    def spline(img1, img2, mask, n=5):
        g_stk1, l_stk1 = get_stack(img1, n)
        g_stk2, l_stk2 = get_stack(img2, n)
        g_mask, l_mask = get_stack(mask, n)

        spline_bands_imgs = []
        for i in range(n):
            mask = g_mask[i] / 255.0
            band_img1 = l_stk1[i]
            band_img2 = l_stk2[i]
            spline_bands_imgs.append(mask * band_img1 + (1-mask) *  band_img2)

        spline_img = spline_bands_imgs[0]
        spline_img = spline_img.astype(np.uint8)
        for i in range(1,n):
            spline_img += spline_bands_imgs[i].astype(np.uint8)
            spline_img = spline_img.astype(np.uint8)
        return spline_img
    

    img2 = cv2.imread('intake/apple.jpeg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('intake/orange.jpeg', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('intake/half_mask.jpg', cv2.IMREAD_GRAYSCALE)

    blended_img = spline(img1, img2, mask)

    cv2.imshow('windows', blended_img)
    cv2.waitKey(0)
    cv2.imwrite('output/multi_resolution_spline.jpg', blended_img)

if __name__ == '__main__':

    #part2_1()
    #part2_2()
    #part2_3()
    part2_4()