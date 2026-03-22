function colorName = analyzeColor(img, stats_i)
    pixels = stats_i.PixelIdxList;
    imgHSV = rgb2hsv(img);
    
    H_chan = imgHSV(:,:,1);
    S_chan = imgHSV(:,:,2);
    V_chan = imgHSV(:,:,3);
    
    H = mean(H_chan(pixels)) * 360;
    S = mean(S_chan(pixels)) * 100;
    V = mean(V_chan(pixels)) * 100;
    
    fprintf('ID Stats: H=%.1f, S=%.1f, V=%.1f | ', H, S, V);

    if V < 22
        colorName = "Black";
    elseif S < 18 && V > 72
        colorName = "White";

    elseif (H > 200 && H < 280) && V < 45
        colorName = "Dark Blue";

    elseif S < 28 && V < 75
        colorName = "Gray";

    elseif (H >= 120 && H <= 145) && S > 30
        colorName = "Green";


    % theroretical HSV ranges are different. 
    % for some reason in this picture those are correct.
    elseif (S > 30 && S < 45) && (V > 58) && (H > 80 && H < 265)
        colorName = "Pink";
    elseif (H > 300 || H < 20) && S < 55
        colorName = "Pink";
        
    elseif (H > 350 || H < 20) && S >= 55
        colorName = "Red";
        
    elseif H >= 20 && H < 70
        colorName = "Yellow";
        
    elseif H >= 70 && H < 165
        colorName = "Green";
        
    elseif H >= 165 && H < 270
        colorName = "Blue";
        
    else
        colorName = "Other";
    end
end