clear; clc; close all;

img = imread('test.jpg'); 

%% Segmentation
% segmentImage exported from Color Thresholder
[binaryMask, maskedRGBImage] = segmentImage(img);

%% Mask filtration
% alternatively, ImageSegmenter could have been used to export a function
% fill internal holes (black numbers inside the shapes)
cleanMask = imfill(binaryMask, 'holes'); 

% closing operation (dilation -> erosion)
% connect fragmented parts and smooth internal gaps
se_close = strel('disk', 10); 
cleanMask = imclose(cleanMask, se_close);

% opening operation (erosion -> dilation)
% remove small noise around the objects and smooth rough edges
se_open = strel('disk', 5);
cleanMask = imopen(cleanMask, se_open);

%% Region analysis 
% alternatively, ImageRegionAnalyzer could have been used to export a function
% extract geometric properties for classification
% added Area to filter out insignificant noise
stats = regionprops(cleanMask, 'BoundingBox', 'Centroid', 'Circularity', 'Area', 'Eccentricity');

imshow(img);
hold on;

%% Classification and annotation
for i = 1:length(stats)
    % filter out small noise that survived morphology
    if stats(i).Area < 500
        continue;
    end
    
    fprintf('Object #%d: Area=%.0f, Circularity=%.3f, Eccentricity=%.3f\n', ...
            i, stats(i).Area, stats(i).Circularity, stats(i).Eccentricity);
    
    if stats(i).Circularity > 0.94
        objectShape = 'Circle';
        boxColor = 'g';
    else
        objectShape = 'Square';
        boxColor = 'r';
    end
    
    rectangle('Position', stats(i).BoundingBox, 'EdgeColor', boxColor, 'LineWidth', 2);
   
    labelStr = sprintf('#%d %s', i, objectShape);
    
    xPos = stats(i).Centroid(1);
    yPos = stats(i).BoundingBox(2) - 20;
    
    text(xPos, yPos, labelStr, ...
        'Color', 'yellow', ...
        'FontSize', 10, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
end
hold off;