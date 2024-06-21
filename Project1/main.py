import numpy as np
import skimage as sk
import cv2  
from skimage import transform


def read_img(path):
    try: 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
    except: 
        print(f"Can't read img from {path}")

def split_img(img):

    height = img.shape[0] // 3
    R_img = img[0:height]
    G_img = img[height:height*2]
    B_img = img[height*2:height*3]
    return R_img, G_img, B_img


def L2_Norm(img1, img2):
    assert img1.shape == img2.shape, 'L2_Norm with different shape imgs'
    return np.linalg.norm(img1 - img2, ord=2, axis=None)

def cut_img(img): # to erase black margin
    cut_height = img.shape[0] // 10 # cut +-10%
    cut_width = img.shape[1] // 10 # cut +-10%

    return img[ cut_height : img.shape[0] - cut_height, cut_width : img.shape[1] - cut_width]

def displacement_search(img1, img2, std, scale, displace_range_x = 2, displace_range_y = 2):
    img1 = transform.rescale(img1, scale)
    img2 = transform.rescale(img2, scale)

    min_norm = 1e18
    best_displacement = std
    for i in range(-int(displace_range_y/scale), int(displace_range_y/scale), 1):
        for j in range(-int(displace_range_x/scale), int(displace_range_x/scale), 1):
            shifted_img1 = np.roll(img1, (std[0] * 2 + i, std[1] * 2 + j), axis=(0,1)) # scale*2, std*2
            norm = L2_Norm(shifted_img1, img2)
            if(norm < min_norm):
                min_norm = norm
                best_displacement = (std[0] * 2 + i, std[1] * 2 + j)
    print(best_displacement, min_norm)
    return best_displacement

def align(img1, img2):
    assert img1.shape == img2.shape, 'align with different shape imgs'
    
    scales = [0.125, 0.25, 0.5, 1.0]
    best_displacement = (0,0)

    if img1.shape[0] + img1.shape[1] > 2000:
        for scale in scales:
            best_displacement = displacement_search(img1, img2, best_displacement, scale)
    else:
        best_displacement = displacement_search(img1, img2, best_displacement, 1.0, 15, 15)

    shifted_img1 = np.roll(img1, (best_displacement[0],best_displacement[1]), axis=(0,1))

    return shifted_img1
    
if __name__ == '__main__':

    img = read_img('intake/arches.tif')
    R_img, G_img, B_img = split_img(img)
    R_img = cut_img(R_img)
    G_img = cut_img(G_img)
    B_img = cut_img(B_img)
    # choose R_img as standard    
    shifted_G_img = align(G_img, R_img)
    shifted_B_img = align(B_img, R_img)
    
    combined_img = np.stack((R_img, shifted_G_img, shifted_B_img), axis=2)
    print(combined_img.shape)
    cv2.imwrite('output.jpg', combined_img)

    cv2.imshow('windows', combined_img)
    cv2.waitKey(0)

    



