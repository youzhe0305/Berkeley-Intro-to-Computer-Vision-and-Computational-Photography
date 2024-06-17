import numpy as np
import cv2
from scipy import signal # signal for process 1-dim signal & 2-dim picture
import matplotlib.pyplot as plt

# To find the edge of photo

def get_fd_x(img): # get finit difference in direction x
    D_x = np.array([[-1,1]])
    return signal.convolve2d(img, D_x, mode='same', boundary='symm') # img, filter, output size same, symmetric padding, 

def get_fd_y(img): # get finit difference in direction y
    D_y = np.array([[-1],[1]])
    return signal.convolve2d(img, D_y, mode='same', boundary='symm') # img, filter, output size same, symmetric padding, 

def get_grad_magnitude(grad_x, grad_y): # design a magnitude for x-y gradient
    return (grad_x**2 + grad_y**2) 

def binarize(grad_mag, threshold): 
    return 255 * (grad_mag > threshold)

def get_gaussian_blur_img(img, ksize=5, sigma=1.0):
    G = cv2.getGaussianKernel(ksize=ksize, sigma=sigma) # Guassian filter, ksize*1, value distribute according to guassian distribution
    G = np.outer(G,np.transpose(G))
    return signal.convolve2d(img, G, boundary='symm', mode='same')

def part1_1(): # directly binarize
    
    cameraman_img = cv2.imread('intake/cameraman.png', cv2.IMREAD_GRAYSCALE)

    grad_x = get_fd_x(cameraman_img)
    grad_y = get_fd_y(cameraman_img)
    fig, axes = plt.subplots(1, 2) # row, cloumn, 
    axes[0].imshow(grad_x)
    axes[1].imshow(grad_y)
    plt.show() 

    grad_mag = get_grad_magnitude(grad_x, grad_y)
    plt.imshow(grad_mag)
    plt.show()  

    edge_img = binarize(grad_mag, 800) # I think 800 is a reasonable threshold (for square sum)
    plt.imshow(edge_img)
    plt.show()
    cv2.imwrite('output/edge_cameraman.png', edge_img)

def part1_2(): # gassian blur 

    img = cv2.imread('intake/cameraman.png', cv2.IMREAD_GRAYSCALE)

    blur_img = get_gaussian_blur_img(img)
    plt.imshow(blur_img)
    plt.show()

    grad_x = get_fd_x(blur_img)
    grad_y = get_fd_y(blur_img)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(grad_x)
    axes[1].imshow(grad_y)
    plt.show()

    grad_mag = get_grad_magnitude(grad_x, grad_y)
    plt.imshow(grad_mag)
    plt.show()

    edge_img = binarize(grad_mag, 800)
    plt.imshow(edge_img)
    plt.show()
    cv2.imwrite('output/edge_gaussain_cameraman.png', edge_img)

def part1_3(): # DoG

    img = cv2.imread('intake/cameraman.png', cv2.IMREAD_GRAYSCALE)
    blur_img1 = get_gaussian_blur_img(img)
    blur_img2 = get_gaussian_blur_img(img, sigma=5.0)
    DoG = blur_img1 - blur_img2
    binary_img = binarize(DoG, 1)
    binary_img_F32 = binary_img.astype(np.float32) 
    cv2.imshow('windows', binary_img_F32) # accuracy too high
    cv2.waitKey(0)
    cv2.imwrite('output/edge_DoG_cameraman.png', binary_img_F32)

if __name__ == '__main__':
    #part1_1()
    #part1_2()
    part1_3()