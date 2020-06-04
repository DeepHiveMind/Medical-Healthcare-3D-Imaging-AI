function [ ] = sort_dataset( mode )
% Input 'mode' can be either 'train' (to sort train images) or 'test' (to sort
% test images)
% Inizialization
main_dir = '/Users/DeepHiveMind/Desktop/matlab_create_segemented_img';

%% Original Images path
img_path = strcat(main_dir,'/',mode);

%% Creation path
creation_path = 'isbi-classification-dataset';
creation_subDir = strcat(main_dir,'/',creation_path);
malignant_path = strcat(creation_subDir,'/',mode,'/malignant');
benign_path = strcat(creation_subDir,'/',mode,'/benign');

files = dir(img_path);    % Run once for 'train_path' and once for 'test_path'
files(1:2) = [];

i = 1;
for image_file = files'
    % get img
    img_file_path = strcat(img_path,'/',image_file.name);
    
    % need to skip over the . file
    img = imread(img_file_path);

    csv_file = strcat(mode,'.csv');
    M = csvread(csv_file);
    if M(i) == 0
        % Image contains benign lesion
        filename = strcat(benign_path,'/',image_file.name);
    else
        % Image contains malignant lesion
        filename = strcat(malignant_path,'/',image_file.name);
    end
    imwrite(img,filename);
    i = i + 1;
end

end
