import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d
from skimage.draw import polygon
import imageio
import skimage.io as skio

# important video:　https://www.youtube.com/watch?v=OCs_KHNpLs8&list=PLS0SUwlYe8czup5qbENhx47-BJ43Zy6It&index=37

def lable_img(path, output_foname, idx): 
    # corner 4
    # head 12
    # eyes(4*2)(left, up, down, right)
    # nose(3) (left, mid, right)
    # mouth(4)(left, up, down, right)
    # ears(3*2)(outside, up, down)
    img = cv2.imread(path)
    plt.imshow(img)
    points = np.array(plt.ginput(n = 37, timeout=0))
    print(points)
    plt.close()
    
    np.save('img_corr_point/' + output_foname + '/' + str(idx) + '.npy', points)

def lable_folder(path, output_foname):
    idx = 1
    for p in os.listdir(path):
        print(f'{path}/{p}')
        lable_img(f'{path}/{p}', output_foname, idx)
        idx += 1

def compute_affin_matrix(tri_pts1, tri_pts2):
    homo_pts1 = np.hstack([ tri_pts1, [[1],[1],[1]] ]).T # make pts from 3*2 matrix to 3*3 matrix
    # [[x1,x2,x3]
    #  [y1,y2,y3]
    #  [1 ,1 ,1]]
    # added constant 1 can be computed as translation(平移)
    homo_pts2 = np.hstack([ tri_pts2, [[1],[1],[1]] ]).T

    T = homo_pts2 @ np.linalg.inv(homo_pts1) # solve T of TA = B, 
    return T 


def apply_wrap(img1, st_pts, ed_pts):

    # plt.imshow(img1)
    # temp = st_pts.T
    # plt.scatter(temp[0], temp[1])
    # plt.show()

    # plt.imshow(img2)
    # temp = ed_pts.T
    # plt.scatter(temp[0], temp[1])
    # plt.show()
    res = np.zeros(img1.shape)
    tri = Delaunay(st_pts)

    st_tri_pts = st_pts[tri.simplices]
    ed_tri_pts = ed_pts[tri.simplices]

    for st_tri_3pt, ed_tri_3pt in zip(st_tri_pts, ed_tri_pts): # 3pt: 3*2 matrix
        transform_matrix = np.linalg.inv(compute_affin_matrix(st_tri_3pt, ed_tri_3pt)) # from ed to st, so invert (T*A=B, T^-1*B=A)

        # 先算變換後的(end)，再把他映射回原圖(st)，把映射到的原圖的地方，貼到變換後的位置
        row = [ i[0] for i in ed_tri_3pt] # x cordinate, row (-)
        column = [ i[1] for i in ed_tri_3pt] # y cordinate, column (|)
        rr, cc = polygon(row,column) # get the polygon(多邊形)'s all the inner pixel (x cordinate array & y cordinate array)

        homo_ed_indices = (np.vstack([rr, cc, np.ones(rr.shape)])).astype(int)

        homo_st_indices = (transform_matrix @ homo_ed_indices).astype(int)
        
        for i in range(len(homo_ed_indices[0])): # [y][x]
            res[homo_ed_indices[1][i]][homo_ed_indices[0][i]] = img1[homo_st_indices[1][i]][homo_st_indices[0][i]]
    return res

def morph(img1, img2, img1_pts, img2_pts, dissolve_frac):

    mid_pts = img1_pts * dissolve_frac + img2_pts * (1-dissolve_frac)
    mid_pts = mid_pts.astype(int)

    mid_img1 = apply_wrap(img1, img1_pts, mid_pts)
    mid_img2 = apply_wrap(img2, img2_pts, mid_pts)

    morphed_img = mid_img1 * dissolve_frac + mid_img2 * (1-dissolve_frac)
    return morphed_img


def plot_triangualtion(fname):
    points = np.load(fname)
    tri = Delaunay(points)
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()  
    print(tri.simplices) # tri.simplices return triangles' vertrex (represented by index of points)

if __name__ == '__main__':

    #lable_folder('intake/01m', '01m') # 要重新標點
    
    img1 = skio.imread('intake/01m/1.jpg')
    img2 = skio.imread('intake/01m/2.jpg')

    st_pts = np.load('img_corr_point/01m/1.npy')
    ed_pts = np.load('img_corr_point/01m/2.npy')

    images = []
    for i in reversed(range(0,10)):
        print(f'GIF Process: {10-i}/{10}')
        alpha = i / 10
        res = morph(img1, img2, st_pts, ed_pts, alpha).astype(np.uint8)
        # temp = res[:,:,0]
        # res[:,:,0] = res[:,:,2]
        # res[:,:,2] = temp
        images.append(res)
        print('done')
    imageio.mimsave('output/mid_way_face.gif', images, duration=0.5)
    pass
    
