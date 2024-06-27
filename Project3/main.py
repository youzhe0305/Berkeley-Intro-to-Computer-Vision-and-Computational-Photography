import matplotlib.pyplot as plt
import cv2
import os

def lable_img(path, output_fname): # eyes(3*2)(left, middle, right), nose(1), mouth(3)(left, middle, right), ears(2)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    points = plt.ginput(n = 12)
    plt.close()
    with open('img_corr_point/' + output_fname, 'a') as f:
        for point in points:
            f.write(str(point[0]))
            f.write(',')
            f.write(str(point[1]))
            f.write('\n')

def lable_folder(path, output_fname):
    for p in os.listdir(path):
        print(f'{path}/{p}')
        lable_img(f'{path}/{p}', output_fname)


if __name__ == '__main__':

    # lable_folder('intake/01m', '01m.txt')
    
    pass
