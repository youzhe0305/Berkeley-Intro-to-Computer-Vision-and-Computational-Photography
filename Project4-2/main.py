
import cv2
import numpy as np


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

def RANSAC(pts1, pts2, threshold=0.1, pick_times = 1000): # pick_times: 要找幾次初始點
    
    print('original points number:', len(pts1))
    best_inlier_idx = []
    most_lnlier_num = 0
    for time in range(0,pick_times):
        # print(f'time {time+1}:')
        random_int = [np.random.randint(0, len(pts1)) for _ in range(4)] # choose 6 points(compute H need at least 4 points)
        inlier_idx = []
        suppose_inlier_pts1 = pts1[random_int]
        suppose_inlier_pts2 = pts2[random_int]
        H_base = compute_H(suppose_inlier_pts1, suppose_inlier_pts2)
        for point_idx in range(0, len(pts1)):
            test_inliner_pts1 = np.append(suppose_inlier_pts1, [pts1[point_idx]], axis=0)
            test_inliner_pts2 = np.append(suppose_inlier_pts2, [pts2[point_idx]], axis=0)
            H_test = compute_H(test_inliner_pts1, test_inliner_pts2)
            error = np.sum((H_base - H_test) ** 2)
            # print('error:', error)
            if error < threshold:
                inlier_idx.append(point_idx)
        # print(inlier_idx)
        if(len(inlier_idx) > most_lnlier_num):
            most_lnlier_num = len(inlier_idx)
            best_inlier_idx = inlier_idx

    print(most_lnlier_num)
    print(best_inlier_idx)
    return best_inlier_idx
    # 待進行...
    pass
    
    

def feature_get(img1, img2):
    
    shift = cv2.SIFT_create() # SIFT dectector
    features1, descriptor1 = shift.detectAndCompute(img1, None) # descirptor: 128維的向量
    features2, descriptor2 = shift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2) 
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]  # 取最好的4個點

    pts1 = np.float32([features1[m.queryIdx].pt for m in good_matches]) # 找出原圖的用來做match的幾個特徵點
    pts2 = np.float32([features2[m.trainIdx].pt for m in good_matches]) # 找出對應圖的特徵點
    inliers_idx = RANSAC(pts1, pts2)
    pts1 = pts1[inliers_idx]
    pts2 = pts2[inliers_idx]
    good_matches = []
    for i in inliers_idx:  
        good_matches.append(matches[i])
    
    np.save('img_corr_point/box/1.npy', pts1)
    np.save('img_corr_point/box/2.npy', pts2)

    # 绘制匹配结果
    img3 = cv2.drawMatches(img1, features1, img2, features2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示图像  
    cv2.imshow('Matches', img3)
    cv2.waitKey(0)
    
    # print(matches)
    
    
    pass
    
    
    
if __name__ == '__main__':
    
    img1 = cv2.imread('box/1.jpg')
    img2 = cv2.imread('box/2.jpg')
    feature_get(img1, img2)