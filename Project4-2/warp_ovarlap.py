import cv2
import cv2.load_config_py2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from skimage import transform

def lable_img(path, output_foname, idx): 
    # 4 or more points
    img = cv2.imread(path)
    plt.imshow(img)
    points = np.array(plt.ginput(n = 6, timeout=0))
    print(points)
    plt.close()    
    np.save('img_corr_point/' + output_foname + '/' + str(idx) + '.npy', points)

def lable_folder(path, output_foname):
    idx = 1
    for p in os.listdir(path):
        print(f'{path}/{p}')
        lable_img(f'{path}/{p}', output_foname, idx)
        idx += 1

def compute_H(pts1, pts2): # correspond points in img1 & img2

    '''
    [wx']       [x]
    [wy'] = H * [y]
    [ w ]       [1]
    
    w = x*h_31 + y*h_32 + h_33
    x' = (x*h_11 + y*h_12 + h_13) / (x*h_31 + y*h_32 + h_33)
    y' = (x*h_21 + y*h_22 + h_33) / (x*h_31 + y*h_32 + h_33)
    
    x*h_11 + y*h_12 + h_13 - x'*x*h_31 - x'*y*h_32 - x'*h_33 = 0
    x*h_11 + y*h_12 + h_13 - y'*x*h_31 - y'*y*h_32 - y'*h_33 = 0

    h_33 only affect scale of value, set h_33 = 1

    [x, y, 1, 0, 0, 0, -x'x, -x'y]   [h_11]   [x']
    [0, 0, 0, x, y, 1, -y'x, -y'y] * [h_12] = [y']
                                     [h_13]   
                                     [h_21]
                                     ...
    
    HA = B, A,B matrix including points
                                     
    '''

    A = []
    B = []
    for pt1, pt2 in zip(pts1, pts2):
        x = pt1[0]
        y = pt1[1]
        xp = pt2[0]
        yp = pt2[1]
        A.append([x, y, 1 ,0, 0, 0, -xp*x, -xp*y])
        A.append([0, 0, 0 ,x, y, 1, -yp*x, -yp*y])
        B.append([xp])
        B.append([yp])
    A = np.array(A)
    B = np.array(B)
    H = np.linalg.lstsq(A, B, rcond=-1)[0]
    H = np.array([[H[0][0], H[1][0], H[2][0]],
                  [H[3][0], H[4][0], H[5][0]],
                  [H[6][0], H[7][0], 1.]])
    return H

def compute_canva_size(img , pts1, pts2):
    
    height = img.shape[0]
    width = img.shape[1]
    print(height, width)
    # (x,y,1)
    lt = np.array([0,0,1]) # left-top, in opencv, plt the coordinate is top->down, left->right (0->inf)
    lb = np.array([0,height,1]) # left-bottom
    rt = np.array([width,0,1]) # right-top
    rb = np.array([width,height,1]) # right-bottom

    H = compute_H(pts1, pts2)

    transformed_lt = H @ lt.T # shape: (3,1)
    transformed_lt /= transformed_lt[2] # normalization

    transformed_lb = H @ lb.T
    transformed_lb /= transformed_lb[2]

    transformed_rt = H @ rt.T
    transformed_rt /= transformed_rt[2]

    transformed_rb = H @ rb.T
    transformed_rb /= transformed_rb[2]

    max_x = math.ceil(max(transformed_rt[0], transformed_rb[0]))
    max_y = math.ceil(max(transformed_lb[1], transformed_rb[1]))
    min_x = math.floor(min(transformed_lt[0], transformed_lb[0]))
    min_y = math.floor(min(transformed_lt[1], transformed_rt[1]))
    canva_width = max_x - min_x + abs(min_x)
    canva_height = max_y - min_y + abs(min_y)

    return canva_width, canva_height

def get_all_points(width, height): # get all pixel in (width, height) square return list = [x,y]
    x_axis = np.arange(0,width,1)
    y_axis = np.arange(0,height,1)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    points = np.stack((x_grid, y_grid), axis=-1)
    points = points.reshape(width*height ,2)
    return points

def warp_img(img1, pts1, pts2):
    
    H = compute_H(pts1, pts2) # may be bug?
    H = np.linalg.inv(H)

    canva_width, canva_height = compute_canva_size(img1, pts1, pts2)
    # print(canva_width, canva_height)
    canva = np.zeros((canva_height, canva_width, 3))
    points = get_all_points(canva_width, canva_height)
    # print(points)
    for pts in points:
        ori_pts = (H @ np.append(pts, 1).T)
        ori_pts = (ori_pts / ori_pts[2]).astype(int)
        if(ori_pts[0] < 0 or ori_pts[1] < 0 or ori_pts[0] >= img1.shape[1] or ori_pts[1] >= img1.shape[0]):
            continue
        # print(pts)
        # print('///')
        # print(ori_pts)
        canva[pts[1],pts[0],:] = img1[ori_pts[1],ori_pts[0],:]
    canva = canva.astype(np.uint8)
    return canva
    cv2.imshow('window', canva)
    cv2.waitKey(0)
    cv2.imwrite('output_temp', canva)

def stack_img(img1, img2, pts1, pts2):

    canva = warp_img(img1, pts1, pts2)
    height = img2.shape[0]
    width = img2.shape[1]
    points = get_all_points(width, height)
    print(canva.shape)
    
    for pts in points:
        if np.equal(canva[pts[1],pts[0],:], np.array([0,0,0])).sum() == 3:
            canva[pts[1],pts[0],:] = img2[pts[1],pts[0],:]
        else:
            canva[pts[1],pts[0],:] = (0.5 * canva[pts[1],pts[0],:] + 0.5 * img2[pts[1],pts[0],:]).astype(np.uint8)
    cv2.imshow('window', canva)
    cv2.waitKey(0)
    cv2.imwrite('output_temp.jpg', canva)

if __name__ == '__main__':

    # lable_folder('desk_high_quality', 'desk_high_quality')
    pts1 = np.load('img_corr_point/box/1.npy')
    pts2 = np.load('img_corr_point/box/2.npy')  
    # print(pts1)
    # print(pts2)
    # H = compute_H(pts1, pts2)
    # print(H)
    # test = np.append(pts1[0], 1.)
    # ans = H @ test.T
    # ans = ans / ans[2]
    # print(ans)

    img1 = cv2.imread('desk/1.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('desk/2.jpg', cv2.IMREAD_COLOR)

    stack_img(img1, img2, pts1, pts2)
    # warp_img(img1, pts1, pts2)
