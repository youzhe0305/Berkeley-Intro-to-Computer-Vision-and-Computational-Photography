import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from scipy.spatial import Delaunay

# important video:　https://www.youtube.com/watch?v=OCs_KHNpLs8&list=PLS0SUwlYe8czup5qbENhx47-BJ43Zy6It&index=37

def lable_img(path, output_foname, idx): 
    # eyes(4*2)(left, up, down, right)
    # nose(3) (left, mid, right)
    # mouth(4)(left, up, down, right)
    # ears(4*2)(left, up, down, right)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    points = np.array(plt.ginput(n = 23, timeout=0))
    plt.close()
    
    np.save('img_corr_point/' + output_foname + '/' + str(idx) + '.npy', points)

def lable_folder(path, output_foname):
    idx = 1
    for p in os.listdir(path):
        print(f'{path}/{p}')
        lable_img(f'{path}/{p}', output_foname, idx)
        idx += 1

def compute_affin_matrix(tri_pts1, tri_pts2):
    homo_pts1 = np.hstack([ tri_pts1, [[1],[1],[1]] ]) # make pts from 3*2 matrix to 3*3 matrix
    # added constant 1 can be computed as translation(平移)
    homo_pts2 = np.hstack([ tri_pts2, [[1],[1],[1]] ])

    return np.linalg.solve(homo_pts1, homo_pts2) # solve the affin problme from matrix1 to marix2

def 

def plot_triangualtion(fname):
    points = np.load(fname)
    tri = Delaunay(points)
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()  
    print(tri.simplices) # tri.simplices return triangles' vertrex (represented by index of points)

if __name__ == '__main__':

    #lable_folder('intake/01m', '01m')
    plot_triangualtion('img_corr_point/01m/1.npy')
    pass
