import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from mid_way_face import apply_wrap



def mean_face_cal(img_folder_path, pts_folder_path):

    exp_img = cv2.imread(f'{img_folder_path}/1a.jpg')
    mean_face = np.zeros(exp_img.shape)
    n_img = len(os.listdir(img_folder_path))
    for p in os.listdir(img_folder_path):
        img = cv2.imread(f'{img_folder_path}/{p}').astype(np.float32)
        mean_face += img * (1/n_img)
    mean_face = mean_face.astype(np.uint8)
    cv2.imshow('windows', mean_face)
    cv2.waitKey(0)
    cv2.imwrite('output/mean_face.jpg', mean_face)

    exp_pts = np.loadtxt(f'{pts_folder_path}/1a.pts', comments=('version:', 'n_points:', '{', '}') )
    mean_pts = np.zeros(exp_pts.shape)
    n_img = len(os.listdir(pts_folder_path))
    for p in os.listdir(pts_folder_path):
        pts = np.loadtxt(f'{pts_folder_path}/{p}', comments=('version:', 'n_points:', '{', '}') )
        mean_pts += pts * (1/n_img)
    
    temp = mean_pts.astype(int).T
    plt.scatter(temp[0], temp[1])
    plt.savefig('output/mean_points.jpg')
    plt.show()

if __name__ == '__main__':
    mean_face_cal('intake/FEI_Face/image', 'intake/FEI_Face/points') # 巴西人臉資料庫
    pass
