clear; clc; close all;
img = imread('test.jpg'); 

%% Segmentation
% segmentImage exported from Color Thresholder
[binaryMask, ~] = segmentImage(img);

%% Mask filtration
% Calling the separate filtration function
cleanMask = filterMask(binaryMask);

%% Region analysis 
% Extracting geometric properties
stats = regionprops(cleanMask, 'BoundingBox', 'Centroid', 'Circularity', 'Area', 'Eccentricity', 'PixelIdxList');

bboxes = [];
labels = {};
boxColors = {};
validCount = 0;

%% Classification and annotation
for i = 1:length(stats)
    % filter out small noise that survived morphology
    if stats(i).Area < 500
        continue;
    end
    
    [objectShape, uiColor] = analyzeShape(stats(i));
    
    objectColor = analyzeColor(img, stats(i));
    
    validCount = validCount + 1;
    bboxes(validCount, :) = stats(i).BoundingBox;
    labels{validCount} = sprintf('#%d %s %s', i, objectColor, objectShape);
    
    boxColors{validCount} = uiColor;
    
    fprintf('Object #%d: Area=%.0f, Circ=%.3f, Color=%s, Shape=%s\n', ...
            i, stats(i).Area, stats(i).Circularity, objectColor, objectShape);
end

if validCount > 0
    finalImg = insertObjectAnnotation(img, 'rectangle', bboxes, labels, ...
        'LineWidth', 3, 'FontSize', 17, 'Color', boxColors, 'TextColor', 'white');
    imshow(finalImg);
else
    imshow(img);
    title('No objects detected');
end