function cleanMask = filterMask(binaryMask)
    % closing operation (dilation -> erosion)
    % connect fragmented parts and smooth internal gaps
    se_close = strel('disk', 7); 
    cleanMask = imclose(binaryMask, se_close);
    
    % opening operation (erosion -> dilation)
    % remove small noise around the objects and smooth rough edges
    se_open = strel('disk', 19);
    cleanMask = imopen(cleanMask, se_open);
end