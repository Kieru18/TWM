function [shapeName, uiColor] = analyzeShape(stats_i)
    if stats_i.Circularity > 0.94
        shapeName = 'Circle';
        uiColor = 'Green';
    else
        shapeName = 'Square';
        uiColor = 'Red';
    end
end