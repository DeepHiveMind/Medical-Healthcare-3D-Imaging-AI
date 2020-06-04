from __future__ import print_function
import os
import numpy as np
import skimage
from skimage import data, draw
from skimage import transform, util
import cv2
data_path = '/imatge/deephivemind/work/image-segmentation/isbi-segmentation-dataset'

image_rows = 420
image_cols = 580

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating train images...')
    print('-'*30)
    for image_name in images:
        print(image_name)
        # Read RGB image
        img = cv2.imread(os.path.join(train_data_path, image_name))
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Rescale to image_rows x image_cols size
        resized_image = cv2.resize(gray_image, (image_cols,image_rows))
	    #img = skimage.io.imread(os.path.join(train_data_path, image_name))
        img = np.array([resized_image])
        imgs[i] = img

        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    print('Saving to .npy files done.')

def create_train_masks_data():
    train_masks_data_path = os.path.join(data_path, 'train_masks')
    images = os.listdir(train_masks_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating train masks images...')
    print('-'*30)
    for image_name in images:
        print(image_name)
        # Read RGB image
        img = cv2.imread(os.path.join(train_masks_data_path, image_name))

        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rescale to image_rows x image_cols size
        resized_image = cv2.resize(gray_image, (image_cols,image_rows))

        img = np.array([resized_image])

        img = np.array([img])

        imgs[i] = img

        print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_mask_train.npy', imgs)
    print('Saving to .npy files done.')

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        print(image_name)
    	#img_id = int(image_name.split('.')[0])
        # Read RGB image
        img = cv2.imread(os.path.join(test_data_path, image_name))
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Rescale to image_rows x image_cols size
        resized_image = cv2.resize(gray_image, (image_cols,image_rows))

        img = np.array([resized_image])

        img = np.array([img])

        imgs[i] = img
        #imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    #np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    #imgs_id = np.load('imgs_id_test.npy')
    #return imgs_test, imgs_id
    return imgs_test

if __name__ == '__main__':
    create_train_data()
    create_train_masks_data()
    create_test_data()
