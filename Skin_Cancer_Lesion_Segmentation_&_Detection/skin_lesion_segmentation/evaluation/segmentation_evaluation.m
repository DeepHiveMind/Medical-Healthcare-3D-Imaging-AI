%% Initialization

directory = 'masks'; %'original' or 'masks'

str_extra_mask = '_Masks.png';


%% loop though mask dir.
files = dir(directory);
i = 1;

for image_file = files'
    % build file paths
    disp(i)
    img_file_path = image_file.name;
    base_file_name = strrep(img_file_path, str_extra_mask, '');

    if base_file_name ~= '.'

        % get img and mask
        img = imread(img_file_path);
        
        % reshape
        img_reshaped = reshape(img',1,46080);
        
        % save to csv file
        filename = strcat('masks_csv/masks_',num2str(i),'.csv');
        csvwrite(filename,img_reshaped')
        
        i = i + 1;
    end

end