function colorName = analyzeColor(img, stats_i)
    pixels = stats_i.PixelIdxList;
    
    imgHSV = rgb2hsv(img);
    
    hChan = imgHSV(:,:,1);
    sChan = imgHSV(:,:,2); 
    vChan = imgHSV(:,:,3);
    
    H = mean(hChan(pixels)) * 360;
    S = mean(sChan(pixels)) * 100;
    V = mean(vChan(pixels)) * 100;
    
    fprintf('ID Stats: H=%.1f, S=%.1f, V=%.1f | ', H, S, V);
    
    if (H > 340 || H < 15) && S > 60 && S < 80 && V > 80 && V < 100
        colorName = "Red";
    elseif (H > 340 || H < 15) && S <= 60 && S > 40 && V > 80 && V < 100
        colorName = "Pink";
    elseif H >= 15 && H < 40 && V < 30
        colorName = "Black";
    elseif H >= 15 && H < 40 && S <= 40 && V > 60 && V < 80
        colorName = "Gray";
    elseif H >= 15 && H < 40 && S <= 40 && V >= 80
        colorName = "White";
    elseif H >= 15 && H < 80 && S > 90
         colorName = "Yellow";
    elseif H > 200 && H < 240 && S > 40 && V > 40
        colorName = "Blue";
    elseif H > 230 && H < 270 && S < 40 && V < 40
        colorName = "Dark Blue";
    elseif H > 60 && H < 110 && S < 20 && V > 60
        colorName = "Light Blue";
    elseif H > 100 && H < 160 && S > 20 && S < 60 && V < 70
        colorName = "Green";
    else
        colorName = "Other";
    end
end