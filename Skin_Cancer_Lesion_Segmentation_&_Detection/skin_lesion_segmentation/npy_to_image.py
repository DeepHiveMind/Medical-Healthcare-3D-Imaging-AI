import numpy as np
from scipy.misc import toimage, imsave
import os

# in_dir = "imgs_train"
# in_dir = "imgs_test"
in_dir = "imgs_mask_test"
# in_dir = "imgs_mask_train"

images_to_convert_path = './' + in_dir + '.npy'

img_array = np.load(images_to_convert_path)

output_dir = 'results/imgs_mask_test'

# these are the selected 'random' 5 images to test
# names will be matched by index for now: 1=22, 2=234...
list_of_random_index = np.arange(379)

#  output directory
def ensure_directory_exist(directory_name):
    exist_bool = os.path.isdir('./' + directory_name)
    if not exist_bool:
        os.mkdir(directory_name)


# use this to show the image rather than save it, if you want
def show_image(image):
    toimage(image).show()


# save the image array to a .png file
def plot_image_save_to_file(name, img_cur):
    #  ensure a directory is present/build if necessary
    save_directory = output_dir  # from global value
    ensure_directory_exist(save_directory)

    #  build full path and save
    file_name = name + '.png'
    full_path = os.path.join(save_directory, file_name)
    imsave(full_path, img_cur)


# convert the numpy array to a int array through .astype('float32')
def convert_numpy_array_to_int_array(img_array):
    print(len(img_array))   # will return number of pictures
    image_list = []
    i = 0
    while i < len(img_array):
        for photo_indiv in img_array[i]:
            image = photo_indiv.astype('float32')
            image_list.append(image)
            # plot_image_save_to_file("jack", image)
            # print(image)
        i += 1
    return image_list


# loop through converted int array and save the image to .png
def convert_int_array_to_png(image_list):
    ind_id = 1
    for photo_array in image_list:
        name = in_dir + '_' + str(ind_id)
        plot_image_save_to_file(name, photo_array)
        ind_id += 1


# create a list of 5 int image array
def get_random_5(img_array_int):
    mySet = set()
    smaller_list = []

    for selected_index in list_of_random_index:
        mySet.add(selected_index)

    i = 0
    while i < len(img_array_int):
        if i in mySet:
            smaller_list.append(img_array_int[i])
        i += 1

    return smaller_list


# wrapper to create 5 'random'(spec. gloablly) .png files to view binary mask
def convert_random_5(img_array_int):
    smaller_list = get_random_5(img_array_int)
    convert_int_array_to_png(smaller_list)


# main wrapper
def main():
    img_array_int = convert_numpy_array_to_int_array(img_array)
    convert_random_5(img_array_int)  # TODO: make sure naming matches
    # convert_all_images(img_array_int) # DONT RUN THIS


main()
