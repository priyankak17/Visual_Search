close all;
clear all;

%% Edit the following line to the folder you unzipped the MSRCv2 dataset to
DATASET_FOLDER = '../MSRC_ObjCategImageDatabase_v2';
%% Folder that holds the results...
DESCRIPTOR_FOLDER = '../descriptors';
%% and within that folder, another folder to hold the descriptors
%% we are interested in working with
%DESCRIPTOR_SUBFOLDER='globalRGBhisto';

DESCRIPTOR_SUBFOLDER='SpatialGridDescriptor';
 % DESCRIPTOR_SUBFOLDER='SpatialGrid_Texture';

% DESCRIPTOR_SUBFOLDER = 'CNN_Descriptor';

% Clean the results folder (search output, PR graphs, confusion matrix, etc)
rmdir_status = rmdir('./results/*', 's');

CATEGORIES = ["Farm Animal" 
    "Tree"
    "Building"
    "Plane"
    "Cow"
    "Face"
    "Car"
    "Bike"
    "Sheep"
    "Flower"
    "Sign"
    "Bird"
    "Book Shelf"
    "Bench"
    "Cat"
    "Dog"
    "Road"
    "Water Features"
    "Human Figures"
    "Coast"
    ];

query_indexes=[301 358 384 436 447 476 509 537 572 5 61 80 97 127 179 181 217 266 276 333];
%query_indexes=[];

%% 1) Load all the descriptors into "ALLFEAT"
%% each row of ALLFEAT is a descriptor (is an image)

ALLFEAT=[];
ALLFILES=cell(1,0);
all_cats=[];
ctr=1;
allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;

    %identify photo category for PR calculation
    split_string = split(fname, '_');
    all_cats(filenum) = str2double(split_string(1));

    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full))./255;
    %img=double(imread(imgfname_full));    %For CNN 
    thesefeat=[];
    featfile=[DESCRIPTOR_FOLDER,'/',DESCRIPTOR_SUBFOLDER,'/',fname(1:end-4),'.mat'];%replace .bmp with .mat
    %load(featfile,'F');

    %Debugging loading file error
    if exist(featfile, 'file')
        load(featfile, 'F');
    else
        disp(['File not found: ', featfile]);
        % Handle the missing file case, possibly by skipping or raising an error
    end

    ALLFILES{ctr}=imgfname_full;
    ALLFEAT=[ALLFEAT ; F];
    ctr=ctr+1;
end

% Initialize variables
cat_hist = histogram(all_cats).Values;
%cat_total = length(cat_hist);
n_img = size(ALLFEAT, 1); % number of images in collection

confusion_matrix = zeros(length(query_indexes));
all_precision = [];
all_recall = [];
ap_values = zeros([1, length(query_indexes)]);

% Iterating over all categories
for iteration = 1:length(query_indexes)
    % Selecting an image as the query
    query_img = query_indexes(iteration);

    % PCA Implementation
    USE_PCA = false;   % change if you want to stop using PCA
    if USE_PCA
        [ALLFEATPCA, E] = Eigen_PCA(ALLFEAT', 'keepf', 0.98);
        ALLFEATPCA = ALLFEATPCA';
    end

    % Compute the distance of each image to the query
    dst = [];
    for i = 1:n_img
        if USE_PCA
            features = ALLFEATPCA;
        else
            features = ALLFEAT;
        end

        candidate = features(i, :);
        query = features(query_img, :);
        the_dst=cvpr_compare(query,candidate); 
        %the_dst = mahalanobis_compare(query, candidate, E);
        category = all_cats(i);
        dst = [dst; [the_dst, i, category]];
    end
    dst = sortrows(dst, 1); % sort the results

    % Calculating PR for each n
    precision_values = zeros([1, n_img - 1]);
    recall_values = zeros([1, n_img - 1]);
    correct_at_n = zeros([1, n_img - 1]);
    query_row = dst(1, :);
    query_category = query_row(3);

    fprintf('Queried category is %s\n', CATEGORIES(query_category));

    dst = dst(2:n_img, :); % skipping the query image

    for i = 1:size(dst, 1)
        rows = dst(1:i, :);
        correct_results = sum(rows(:, 3) == iteration);
        precision = correct_results / i;
        recall = correct_results / (cat_hist(iteration) - 1);
        precision_values(i) = precision;
        recall_values(i) = recall;
        correct_at_n(i) = rows(i, 3) == iteration;
    end

    % Calculate AP
    average_precision = sum(precision_values .* correct_at_n) / cat_hist(iteration);
    ap_values(iteration) = average_precision;

    all_precision = [all_precision; precision_values];
    all_recall = [all_recall; recall_values];


    if ~exist(strcat('./results', '/', CATEGORIES(dst(1, 3)), '_', string(query_img)), 'dir')
        mkdir(strcat('./results', '/', CATEGORIES(dst(1, 3)), '_', string(query_img)))
    end

    
    % Visualize results and populate confusion matrix
    SHOW=25; % Show top 25 results
    dst=dst(1:SHOW,:);
    outdisplay=[];
    for i=1:size(dst,1)
        img=imread(ALLFILES{dst(i,2)});
        img=img(1:2:end,1:2:end,:); % make image a quarter size
        img=img(1:81,:,:); % crop image to uniform size vertically (some MSVC images are different heights)
        outdisplay=[outdisplay img];

        %populate confusion matrix
        if length(query_indexes)>1
            % figure;
            confusion_matrix(dst(i,3), iteration) = confusion_matrix(dst(i,3), iteration) + 1;
           
        
            
            
        end
    end

  
end

   % Normalize confusion matrix
    norm_confusion_matrix = confusion_matrix ./ sum(confusion_matrix, 'all');
    cm = confusionchart(confusion_matrix, CATEGORIES, 'Normalization', 'column-normalized');
            cm.Title = 'Spatial Colour and Texture Confusion Matrix without PCA (4x3, 5 bins)';
            xlabel('Query Classification');
            ylabel('Ground Truth');
            % Calculate MAP
            figure(4)
            histogram(ap_values);
            title('Average Precision Distribution');
            ylabel('Count');
            xlabel('Average Precision');
            xlim([0, 1]);

% Plotting cumulative PR curve
    %pr_graph = figure('Visible','off');
    figure;
    plot(all_recall, all_precision);
    hold on;
    title('PR Curve');
    xlabel('Recall');
    ylabel('Precision');
    
    %saveas(pr_graph, strcat('./results', '/', CATEGORIES(dst(1, 3)), '_', string(query_img), '/', 'pr_graph'), 'jpg');
    hold off;

% Plot average PR curve
figure(4)
mean_precision = mean(all_precision);
mean_recall = mean(all_recall);
plot(mean_recall, mean_precision, 'LineWidth', 5);
title('Spatial Colour and Texture Average PR without PCA (4x3, 5 bins)');
xlabel('Average Recall');
ylabel('Average Precision');
xlim([0 1]);
ylim([0 1]);



map = mean(ap_values);
disp(map);
ap_sd = std(ap_values);

% Visualization of top results
SHOW=15; % Show top 15 results
dst=dst(1:SHOW,:);
outdisplay=[];
for i=1:size(dst,1)
   img=imread(ALLFILES{dst(i,2)});
   img=img(1:2:end,1:2:end,:); % make image a quarter size
   img=img(1:81,:,:); % crop image to uniform size vertically (some MSVC images are different heights)
   outdisplay=[outdisplay img];
end

figure(5)
imshow(outdisplay);
axis off;

% % Search output
search_output = figure('Visible','off');
montage(ALLFILES(dst(:,2)), 'Size', [1 NaN]);
saveas(search_output, strcat('./results', '/', CATEGORIES(dst(1, 3)), '_', string(query_img), '/', 'search_output'), 'jpg');

