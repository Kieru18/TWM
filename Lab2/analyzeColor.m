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
    
    if S>75
        colorName="Yellow";
    elseif H<15
        colorName="Red";
    elseif (H>34 && H<36) || (H>52 && H<56)
        colorName="White";
    elseif (H>30 && H<34)
        colorName="Gray";
    elseif (H>25 && H<30)
        colorName="Black";
    elseif S<14
        colorName="Light Blue";
    elseif (S>15 && S<30)
        colorName="Dark Blue";
    elseif (V<35)
        colorName="Black";
    elseif (S>45)
        colorName="Blue";
    elseif (S>35 && V>=64.5) || (V>64.9)
        colorName="Pink";
    elseif (H>120 && H<140)
        colorName="Green";
    elseif (H>150 && H<190)
        colorName="Light Blue";
    else
        colorName = "Other";
    end
end