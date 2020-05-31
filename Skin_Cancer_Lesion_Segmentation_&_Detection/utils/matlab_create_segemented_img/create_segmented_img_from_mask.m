%% Initialization
main_directory = 'isbi-segmentation-dataset';     % parent directory

% sub directory
img_subDir = 'test_data';
mask_subDir = 'test_masks';
segmented_subDir = 'segmented_train';    % IMPORTANT: Change either 'segmented_train' and 'segmented_test'
                                        % to download the segmented images in that path name.
malignant_subDir = 'malignant';
benign_subDir = 'benign';
img_directory_path = strcat(main_directory, '/', img_subDir);
mask_directory_path = strcat(main_directory, '/', mask_subDir);
segmented_directory_path = strcat(main_directory, '/', segmented_subDir, '/');

photo_ext = '.jpg';          % image type
mask_ext = '.png';          % image type

individual_file = '*';      % set to * for all files in a folder

% get base file name
str_to_strip_from_mask = '_Segmentation';
str_to_strip_from_img = '';

% full basename + ___ for maks and img
str_extra_mask = strcat(str_to_strip_from_mask, mask_ext);
str_extra_img = strcat(str_to_strip_from_img, photo_ext);

%% loop though mask dir.
mask_files = dir(mask_directory_path);
i = 1;
for image_file = mask_files'
    % build file paths
    mask_file_path = image_file.name;
    base_file_name = strrep(mask_file_path, str_extra_mask, '');
    img_file_path = strcat(base_file_name, str_to_strip_from_img, photo_ext);

    % confirm the paths are correct
    % DEBUGING: sprintf('[%s]: %s -> %s',base_file_name, mask_file_path, img_file_path)

    % need to skip over the . file
    if base_file_name ~= '.'
        % DEBUGING: confirm individual path is correct
        % sprintf('[%s]: %s -> %s',base_file_name, mask_file_path, img_file_path)

        % get img and mask
        img = imread(img_file_path);
        mask = imread(mask_file_path);
        % 100 is an arbitrary value between 0 and 255 (the current values)

        % create mask as logical
        mask_bw = logical(mask>100);

        % this may need

        % function found here: https://www.mathworks.com/matlabcentral/answers/38547-masking-out-image-area-using-binary-mask
        % Inew = img.*repmat(mask,[1,1,3]);

        % try to do it by channel...
%         % get each channel
%         r = img(:,:,1);
%         g = img(:,:,2);
%         b = img(:,:,3);
%
%         % apply logical mask to each channel
% %         r = r(mask_bw);
% %         g = g(mask_bw);
% %         b = b(mask_bw);
%
%         % reconstruct the rgb image
%         segmented_img(:,:,1) = r;
%         segmented_img(:,:,2) = g;
%         segmented_img(:,:,3) = b;

        % this function was adopted from: https://www.mathworks.com/matlabcentral/answers/2646-image-segmentation
        maskedRgbImage = bsxfun(@times, img, cast(mask_bw,class(img)));


        % you will need to add logic for saving the image here after you
        % are sure it's in right format
        csv_file = 'test.csv';
        M = csvread(csv_file);
        if M(i) == 0
            % Image contains benign lesion
            download_directory_path = strcat(segmented_directory_path,benign_subDir,'/',base_file_name);
        else
            % Image contains malignant lesion
            download_directory_path = strcat(segmented_directory_path,malignant_subDir,'/',base_file_name);
        end
        filename = strcat(download_directory_path,photo_ext);
        imwrite(maskedRgbImage,filename)
        i = i + 1;
    end

end

%% show the last image to make sure it's working
figure
subplot(1,3,1)
imshow(img);
title(base_file_name)

subplot(1,3,2)
imshow(mask);
title('Mask')

subplot(1,3,3)
imshow(maskedRgbImage);
title('segmented')
