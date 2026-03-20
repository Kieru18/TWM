function cleanMask = filterMask(binaryMask)
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
end