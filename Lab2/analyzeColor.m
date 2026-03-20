function colorName = analyzeColor(img, stats_i)
    pixels = stats_i.PixelIdxList;
    r = mean(img(pixels));
    g = mean(img(pixels + numel(img)/3));
    b = mean(img(pixels + 2*numel(img)/3));

    if r > g && r > b && g > (r * 0.5)
        colorName = 'Yellow';
    elseif r > g && r > b
        colorName = 'Red';
    elseif g > r && g > b
        colorName = 'Green';
    elseif b > r && b > g
        colorName = 'Blue';
    else
        colorName = 'Object';
    end
end