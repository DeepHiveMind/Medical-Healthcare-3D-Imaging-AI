### Convert a float 32 image into a binary image by thresholding
import numpy as np
import cv2
import os

directory = './results/imgs_mask_test/'
output_directory = './results/binary_imgs_mask_test/'

i = 0

for filename in os.listdir(directory):
    mask = str(directory) + str(filename)
    img = cv2.imread(mask)
    print (mask)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    binary_image = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
    file_name = 'binary_mask_' + str(i) + '.png'
    outfile = output_directory + file_name
    cv2.imwrite(outfile, binary_image)
    i = i + 1

    # make a 1-dimensional view of arr
    flat_arr = arr.ravel()
