function [shapeName, uiColor] = analyzeShape(stats_i)
    if stats_i.Circularity >= 0.957
        shapeName = 'Circle';
        uiColor = 'Green';
    else if stats_i.Eccentricity < 0.85
        shapeName = 'Square';
        uiColor = 'Red';
    else
        shapeName = 'Other';
        uiColor = 'Black';
    end
end