%% Inicjalizacja

vidReader = VideoReader('visiontraffic.avi');
opticFlow = opticalFlowFarneback;

% Pominięcie klatek statycznych na początku.
for i = 1:90
    frame = readFrame(vidReader);
end

% Inicjalizacja estymatora przepływu ostatnią pominiętą klatką
% (eliminuje szum na pierwszej klatce roboczej).
frameGray = rgb2gray(frame);
estimateFlow(opticFlow, frameGray);

ba = vision.BlobAnalysis;

figure(1); clf;
tiledlayout(2, 2, 'Padding', 'none', 'TileSpacing', 'compact');

%% Parametry śledzenia

MAX_ASSOC_DIST = 80;   % [px]    max odległość centroidu przy kojarzeniu
MAX_INVISIBLE  = 8;    % [kl.]   po tylu klatkach bez detekcji track jest zamykany
MAX_TRAIL      = 60;   % [pkt]   max długość wyświetlanej ścieżki
MIN_DRAW_PTS   = 2;    % [pkt]   min liczba punktów do narysowania linii

%% Stan śledzenia

tracks        = [];    % tablica struktur pojazdów (id, centroids, color, ...)
nextTrackID   = 1;
frameIdx      = 0;
totalVehicles = 0;

% Paleta 10 wyraźnie różnych kolorów dla ścieżek.
palette = [
    1.00  0.20  0.20;   % czerwony
    0.20  0.85  0.20;   % zielony
    0.30  0.50  1.00;   % niebieski
    1.00  0.80  0.10;   % żółty
    1.00  0.45  0.00;   % pomarańczowy
    0.85  0.20  0.85;   % różowy/fuksja
    0.15  0.85  0.85;   % cyjan
    0.90  0.60  0.30;   % brązowy
    0.60  0.10  0.60;   % fioletowy
    0.10  0.65  0.45;   % morski
];
nPalette = size(palette, 1);

%% Pętla główna

while hasFrame(vidReader)
    frameIdx = frameIdx + 1;

    frameRGB  = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);

    %----------------------------------------------------------------------
    % 1. Przepływ optyczny
    %----------------------------------------------------------------------
    flow   = estimateFlow(opticFlow, frameGray);
    spdMap = flow.Magnitude;
    dirMap = flow.Orientation;

    nexttile(1); cla;
    imshow(frameRGB * 0.3); hold on;
    plot(flow, 'DecimationFactor', [15 15], 'ScaleFactor', 5);
    hold off;
    title('Wektory przepływu');

    nexttile(2);
    imshow(spdMap, [0, 10]);
    colormap(gca, 'jet');
    title('Mapa prędkości');

    %----------------------------------------------------------------------
    % 2. Progowanie i analiza obszarów
    %----------------------------------------------------------------------
    thr = spdMap > 2;
    thr = imclose(thr, strel('disk', 5));   % wypełnienie drobnych dziur

    nexttile(3);
    imshow(thr);
    title('Maska obszarów ruchu');

    filtDir = zeros(size(dirMap));
    filtDir(thr) = dirMap(thr);

    [AREA, CENTROID, BBOX] = step(ba, thr);

    %----------------------------------------------------------------------
    % 3. Budowanie listy detekcji (obszary > 2000 px)
    %----------------------------------------------------------------------
    detections = [];
    for i = 1:size(AREA, 1)
        if AREA(i) > 2000
            % BUG FIX: oryginał używał x,y,w,h niezadeklarowanych w pętli
            x = BBOX(i, 1);  y = BBOX(i, 2);
            w = BBOX(i, 3);  h = BBOX(i, 4);

            dirPatch = filtDir(y:y+h-1, x:x+w-1);
            spdPatch = spdMap(y:y+h-1, x:x+w-1);

            avgDir = mean(dirPatch(dirPatch ~= 0));
            avgSpd = mean(spdPatch(spdPatch > 2));
            if isnan(avgDir), avgDir = 0; end
            if isnan(avgSpd), avgSpd = 0; end

            det.bb  = BBOX(i, :);
            det.dir = avgDir;
            det.spd = avgSpd;
            det.cc  = CENTROID(i, :);
            det.lbl = AREA(i);

            detections = [detections, det]; %#ok<AGROW>
        end
    end

    nDet = length(detections);
    nTrk = length(tracks);

    %----------------------------------------------------------------------
    % 4. Kojarzenie detekcji ze ścieżkami — zachłanne nearest-neighbour
    %
    %   Macierz odległości distMat(t,d) = odległość euklidesowa między
    %   ostatnią pozycją ścieżki t a centroidem detekcji d.
    %   Pary (t,d) sortowane rosnąco; każda ścieżka i detekcja może
    %   zostać skojarzona co najwyżej raz.
    %----------------------------------------------------------------------
    assigned      = false(1, nDet);
    assignedTrack = zeros(1,  nDet);

    if nTrk > 0 && nDet > 0
        distMat = inf(nTrk, nDet);
        for t = 1:nTrk
            if ~tracks(t).active, continue; end
            lastPos = tracks(t).centroids(end, :);
            for d = 1:nDet
                distMat(t, d) = norm(lastPos - detections(d).cc);
            end
        end

        [vals, sortIdx] = sort(distMat(:));
        usedTrk = false(1, nTrk);
        usedDet = false(1, nDet);
        for k = 1:numel(vals)
            if vals(k) > MAX_ASSOC_DIST, break; end
            [t, d] = ind2sub([nTrk, nDet], sortIdx(k));
            if usedTrk(t) || usedDet(d), continue; end
            assignedTrack(d) = t;
            assigned(d)      = true;
            usedTrk(t)       = true;
            usedDet(d)       = true;
        end
    end

    %----------------------------------------------------------------------
    % 5. Aktualizacja skojarzonych ścieżek
    %----------------------------------------------------------------------
    for d = 1:nDet
        if assigned(d)
            t = assignedTrack(d);
            tracks(t).centroids = [tracks(t).centroids; detections(d).cc];
            tracks(t).lastSeen  = frameIdx;
        end
    end

    %----------------------------------------------------------------------
    % 6. Nowe ścieżki dla nieskojarzonych detekcji → nowy pojazd
    %----------------------------------------------------------------------
    for d = 1:nDet
        if ~assigned(d)
            newTrack.id        = nextTrackID;
            newTrack.centroids = detections(d).cc;       % wiersz [x, y]
            newTrack.color     = palette(mod(nextTrackID-1, nPalette)+1, :);
            newTrack.lastSeen  = frameIdx;
            newTrack.active    = true;

            if isempty(tracks)
                tracks = newTrack;
            else
                tracks(end+1) = newTrack; %#ok<AGROW>
            end

            nextTrackID   = nextTrackID + 1;
            totalVehicles = totalVehicles + 1;
        end
    end

    %----------------------------------------------------------------------
    % 7. Zamykanie ścieżek utraconych przez MAX_INVISIBLE klatek
    %----------------------------------------------------------------------
    for t = 1:length(tracks)
        if tracks(t).active && (frameIdx - tracks(t).lastSeen) > MAX_INVISIBLE
            tracks(t).active = false;
        end
    end

    %----------------------------------------------------------------------
    % 8. Wizualizacja — tile 4
    %----------------------------------------------------------------------
    ann = frameRGB;
    if nDet > 0
        lblCell = arrayfun(@(a) sprintf('%d px', a), ...
                           vertcat(detections.lbl), 'UniformOutput', false);
        ann = insertObjectAnnotation(ann, 'rectangle', ...
              vertcat(detections.bb), lblCell, ...
              'TextBoxOpacity', 0.9, 'FontSize', 18);
    end

    nexttile(4); cla;
    imshow(ann);
    hold on;

    % Rysowanie ścieżek wszystkich pojazdów (aktywnych i zamkniętych)
    for t = 1:length(tracks)
        pts = tracks(t).centroids;
        col = tracks(t).color;

        % Ogranicz wyświetlaną historię do MAX_TRAIL ostatnich punktów
        if size(pts, 1) > MAX_TRAIL
            pts = pts(end-MAX_TRAIL+1:end, :);
        end

        % Linia łącząca kolejne pozycje centroidu
        if size(pts, 1) >= MIN_DRAW_PTS
            plot(pts(:,1), pts(:,2), '-', 'Color', col, 'LineWidth', 2);
        end

        % Marker ostatniej (lub bieżącej) pozycji
        plot(pts(end,1), pts(end,2), 'o', ...
             'Color', col, 'MarkerFaceColor', col, 'MarkerSize', 8);

        % Etykieta z numerem pojazdu
        text(pts(end,1)+6, pts(end,2)-6, sprintf('#%d', tracks(t).id), ...
             'Color', col, 'FontSize', 9, 'FontWeight', 'bold', ...
             'BackgroundColor', [0 0 0]);
    end

    % Wektory średniego kierunku dla bieżących detekcji
    for i = 1:nDet
        det = detections(i);
        x1 = det.cc(1);  y1 = det.cc(2);
        x2 = x1 + 5 * det.spd * cos(det.dir);
        y2 = y1 + 5 * det.spd * sin(det.dir);
        line([x1 x2], [y1 y2], 'Color', 'w', 'LineWidth', 2);
    end

    % Licznik w tytule
    if ~isempty(tracks)
        nActive = sum([tracks.active]);
    else
        nActive = 0;
    end
    title(sprintf('Łącznie: %d pojazd(ów)  |  Aktywne: %d', totalVehicles, nActive));
    hold off;

    drawnow;
    pause(0.01);
end

%% Podsumowanie

fprintf('\n==============================================\n');
fprintf('  Łączna liczba pojazdów na nagraniu: %d\n', totalVehicles);
fprintf('==============================================\n');
