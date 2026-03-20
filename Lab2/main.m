clear; clc; close all;
img = imread('nictelefon2.jpg'); 

%% Segmentation
% segmentImage exported from Color Thresholder
[binaryMask, ~] = createMask(img);

%% Mask filtration
cleanMask = segmentImage(img, binaryMask);

%% Region analysis 
stats = regionprops(cleanMask, 'BoundingBox', 'Centroid', 'Circularity', 'Area', 'Eccentricity', 'PixelIdxList', 'EquivDiameter');

annotatedImg = img;

%% Classification and annotation
for i = 1:length(stats)
    if stats(i).Area < 550
        continue;
    end
    
    [objectShape, uiColor] = analyzeShape(stats(i));
    objectColor = analyzeColor(img, stats(i));
    
    labelStr = sprintf('#%d %s %s', i, objectColor, objectShape);
    
    if strcmp(objectShape, 'Circle')
        shapeType = 'circle';
        coords = [stats(i).Centroid, stats(i).EquivDiameter/2];
    else
        shapeType = 'rectangle';
        coords = stats(i).BoundingBox;
    end
    
    annotatedImg = insertObjectAnnotation(annotatedImg, shapeType, coords, labelStr, ...
        'LineWidth', 3, 'FontSize', 30, 'Color', uiColor, 'TextColor', 'white');
    
    fprintf('Object #%d: Area=%.0f, Circ=%.3f, Color=%s, Shape=%s\n', ...
            i, stats(i).Area, stats(i).Circularity, objectColor, objectShape);
end

figure;
imshow(annotatedImg);