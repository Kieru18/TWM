%% =========================================================
%  Detektor sylwetek – HOG + SVM (sliding window)
%  Jakub Kieruczenko
%  Franciszek Malewski
%
%  Sekcje:
%    1. Konfiguracja
%    2. Wczytywanie danych treningowych
%    3. Ekstrakcja cech HOG
%    4. Trening SVM
%    5. Wczytywanie danych testowych i anotacji
%    6. Detekcja (piramida skali + sliding window + NMS)
%    7. Ewaluacja i wizualizacja (P-R curve, mAP)
%
%  Struktura katalogów:
%    data/pos/                                 – pozytywne próbki treningowe (64×128)
%    data/neg/                                 – negatywne próbki treningowe (64×128)
%    data/test/INRIAPerson/Train/neg/          - negatywne próbki treningowe
%                                                pełnoskalowe
%    data/test/INRIAPerson/Test/pos/           – obrazy testowe
%    data/test/INRIAPerson/Test/annotations/   – pliki .txt z round Truth (format INRIA/PASCAL)
%    data/test/INRIAPerson/Test/pos/gTruth.mat - GT obrazów testowych z lab
%% =========================================================

%% 1. KONFIGURACJA
clear; clc; close all;
rng(290);

% Parametry okna
WIN_W = 64;
WIN_H = 128;

% Parametry sliding window
STEP = 8;
SCALES = 1 ./ (1.1 .^ (0:20));

% Parametry treningu
SVM_C       = 1.85;   % parametr regularyzacji SVM (Box Constraint)
CROP_PER_IMAGE = 20;  % liczba wycinków z obrazów klasy negatywnej

% Parametry detekcji i ewaluacji
SVM_THR     = 3.;   % próg wyniku SVM
NMS_IOU_THR = 0.7;    % próg IoU dla Non-Maximum Suppression
EVAL_IOU    = 0.3;    % próg IoU dla TP/FP w ewaluacji
TEST_N      = 8;      % maksymalna liczba testowych obrazów

% Ścieżki do danych
DIR_POS = fullfile('data', 'pos');
%DIR_NEG = fullfile('data', 'neg');
DIR_NEG = fullfile('data', 'test', 'INRIAPerson', 'Train', 'neg');
DIR_TEST_IMGS = fullfile('data', 'test', 'INRIAPerson', 'Test', 'pos');
DIR_TEST_ANN  = fullfile('data', 'test', 'INRIAPerson', 'Test', 'annotations');

GTRUTH_PATH = fullfile('data', 'test', 'INRIAPerson', 'Test', 'pos', 'gTruth.mat');

fprintf('=== Konfiguracja OK ===\n');
fprintf('  Okno:    %dx%d  |  Krok: %d  |  Skale: %d\n', ...
    WIN_W, WIN_H, STEP, numel(SCALES));
fprintf('  SVM C=%.8f\ndd', SVM_C); 

%% 2. WCZYTYWANIE DANYCH TRENINGOWYCH

pos_files = collect_images(DIR_POS);
neg_files = collect_images(DIR_NEG);

assert(~isempty(pos_files), 'Brak plików klasy pozytywnej');
assert(~isempty(neg_files), 'Brak plików klasy negatywnej');

fprintf('[Dane] Próbki pozytywne : %d obrazów\n', numel(pos_files));
fprintf('[Dane] Próbki negatywne : %d obrazów\n', numel(neg_files));

%% 3. EKSTRAKCJA CECH HOG

hog_len = numel(extractHOGFeatures(zeros(WIN_H, WIN_W, 'uint8')));
fprintf('\n[HOG] Długość wektora: %d\n', hog_len);

% --- Próbki pozytywne ---
fprintf('[HOG] Ekstrakcja – próbki pozytywne...\n');
X_pos = zeros(numel(pos_files), hog_len, 'single');
for k = 1:numel(pos_files)
    img = load_gray_resized( ...
        fullfile(pos_files(k).folder, pos_files(k).name), WIN_H, WIN_W);
    X_pos(k,:) = extractHOGFeatures(img);
end
fprintf('       Gotowe: %d wektorów\n', size(X_pos,1));

% --- Próbki negatywne ---
fprintf('[HOG] Ekstrakcja – próbki negatywne...\n');

total_neg_windows = numel(neg_files) * CROP_PER_IMAGE;

X_neg = zeros(total_neg_windows, hog_len, 'single');
neg_idx = 0;

for k = 1:numel(neg_files)
    img_path = fullfile(neg_files(k).folder, neg_files(k).name);
    full_img = imread(img_path);
    
    if size(full_img, 3) == 3
        full_img = im2gray(full_img);
    end
    
    [h_img, w_img] = size(full_img);
    
    if h_img < WIN_H || w_img < WIN_W
        continue;
    end
    
    for c = 1:CROP_PER_IMAGE
        rand_y = randi(h_img - WIN_H + 1);
        rand_x = randi(w_img - WIN_W + 1);
        
        sub_win = full_img(rand_y : rand_y + WIN_H - 1, rand_x : rand_x + WIN_W - 1);
        
        neg_idx = neg_idx + 1;
        X_neg(neg_idx, :) = extractHOGFeatures(sub_win);
    end
end

X_neg = X_neg(1:neg_idx, :);
fprintf('       Gotowe: %d wektorów z %d obrazów tła\n', size(X_neg,1), numel(neg_files));

% --- Złożenie zbioru treningowego ---
X_train = [X_pos; X_neg];
Y_train = [ones(size(X_pos,1), 1); zeros(size(X_neg,1), 1)];

fprintf('\n[Dane] Zbiór treningowy: %d pos / %d neg  (razem %d)\n\n', ...
    size(X_pos,1), size(X_neg,1), size(X_train,1));

%% 4. TRENING SVM

fprintf('[SVM] Trening klasyfikatora (kernel: linear, C=%.8f)...\n', SVM_C);

svm_mdl = fitcsvm(X_train, Y_train, ...
    'KernelFunction', 'linear', ...
    'BoxConstraint', SVM_C, ...
    'Standardize',     false, ...
    'ClassNames',      [0; 1]);


[labels, raw_scores] = predict(svm_mdl, X_train);
pos_class_col = svm_mdl.ClassNames == 1;
train_scores = raw_scores(:, pos_class_col);

global_min = min(train_scores);
global_max = max(train_scores);
fprintf('Zakres ocen: [%.2f, %.2f]\n', global_min, global_max);

% svm_mdl = fitPosterior(svm_mdl);
% [~, post_scores] = predict(svm_mdl, X_train);
% fprintf('Zakres ocen po fitPosterior: [%.2f, %.2f]\n', min(post_scores(:,2)), max(post_scores(:,2)));

fprintf('[SVM] Trening zakończony.\n');

% Walidacja krzyżowa (5-fold)
%fprintf('[SVM] Walidacja krzyżowa 5-fold...\n');
%cv_mdl  = crossval(svm_mdl, 'KFold', 5);
%cv_loss = kfoldLoss(cv_mdl);
%fprintf('[SVM] Błąd CV: %.2f%%\n\n', cv_loss * 100);

% Indeks kolumny odpowiadającej klasie "osoba" (1) w macierzy score
pos_class_col = find(svm_mdl.ClassNames == 1);

%% 5. WCZYTYWANIE DANYCH TESTOWYCH I ANOTACJI
%test_files = collect_images(DIR_TEST_IMGS);

%dir_path = './data/test/INRIAPerson/Test/pos/';
test_files = collect_test_images(DIR_TEST_IMGS);

assert(~isempty(test_files), 'Brak obrazów testowych w %s', DIR_TEST_IMGS);

% Ogranicz do TEST_N pierwszych obrazów
if numel(test_files) > TEST_N
    test_files = test_files(1:TEST_N);
end
fprintf('[Test] Obrazy testowe: %d\n', numel(test_files));

% Wczytaj anotacje ground truth
GTRUTH_FILENAMES = {'people_1', 'people_2', 'people_3', 'people_4'};
gt_mat_map = load_gtruth_mat(GTRUTH_PATH, GTRUTH_FILENAMES);
gt_boxes = cell(numel(test_files), 1);
for k = 1:numel(test_files)
    [~, fname, ~] = fileparts(test_files(k).name);
    ann_path = fullfile(DIR_TEST_ANN, [fname '.txt']);
    if isfile(ann_path)
        gt_boxes{k} = parse_pascal_annotation(ann_path);
        fprintf('  [GT] %s : %d osób\n', fname, size(gt_boxes{k},1));
    elseif isKey(gt_mat_map, fname)
        gt_boxes{k} = gt_mat_map(fname);
        fprintf('  [GT] %s : %d osób\n', fname, size(gt_boxes{k},1));
    else
        gt_boxes{k} = zeros(0, 4);
        fprintf('  [GT] %s : brak pliku anotacji\n', fname);
    end
end
fprintf('\n');

%% 6. DETEKCJA (PIRAMIDA SKALI + SLIDING WINDOW + NMS)
all_dets   = cell(numel(test_files), 1);
all_scores = cell(numel(test_files), 1);

for img_idx = 1:numel(test_files)
    img_path = fullfile(test_files(img_idx).folder, test_files(img_idx).name);
    It       = imread(img_path);
    It_gray  = im2gray(It);

    dets   = zeros(0, 4);
    scores = zeros(1, 0);

    fprintf('[Det] Obraz %2d/%d : %s\n', img_idx, numel(test_files), ...
        test_files(img_idx).name);

    for si = 1:numel(SCALES)
        scale   = SCALES(si);
        cur_img = imresize(It_gray, scale);
        [H, W]  = size(cur_img);

        count_x = floor((W - WIN_W) / STEP);
        count_y = floor((H - WIN_H) / STEP);
        if count_x < 1 || count_y < 1, continue; end

        % ---- Zbierz cechy HOG wszystkich okien w bieżącej skali ----
        n_win    = count_x * count_y;
        W_batch  = zeros(n_win, hog_len, 'single');
        coords   = zeros(n_win, 2);   % [j (row), i (col)] indeksy siatki okien
        idx = 0;
        for j = 0:count_y-1
            for i = 0:count_x-1
                x   = 1 + i*STEP;
                y   = 1 + j*STEP;
                sub = cur_img(y:y+WIN_H-1, x:x+WIN_W-1);
                idx = idx + 1;
                W_batch(idx,:)  = extractHOGFeatures(sub);
                coords(idx,:)   = [j, i];
            end
        end

        % ---- Jeden zbiorczy predict dla całej skali ----
        [~, score_mat] = predict(svm_mdl, W_batch);
        svm_score = score_mat(:, pos_class_col);

        % Normalizacja
        %svm_score = (svm_score - global_min) / (global_max - global_min);
        %svm_score = max(0, min(1, svm_score));

        % Zachowaj okna powyżej progu
        keep = svm_score > SVM_THR;
        pos_idx = find(keep);

        for pi = 1:numel(pos_idx)
            ii = pos_idx(pi);
            j  = coords(ii, 1);
            i  = coords(ii, 2);
            % Przelicz współrzędne z powrotem do skali oryginalnej
            x = (1 + i*STEP) / scale;
            y = (1 + j*STEP) / scale;
            w = WIN_W / scale;
            h = WIN_H / scale;
            dets   = [dets;   x, y, w, h];               %#ok<AGROW>
            scores = [scores, svm_score(ii)'];           %#ok<AGROW>
        end
    end

    fprintf('        Kandydatów przed NMS: %d\n', size(dets,1));

    % Non-Maximum Suppression
    [dets, scores] = nms_filter(dets, scores, NMS_IOU_THR);
    fprintf('        Po NMS: %d detekcji\n', size(dets,1));

    all_dets{img_idx}   = dets;
    all_scores{img_idx} = scores;

    % ---- Wizualizacja ground truth i predykcji ----
    visualize_result(It, dets, scores, gt_boxes{img_idx}, img_idx, ...
        test_files(img_idx).name);
end

%% 7. EWALUACJA I WIZUALIZACJA P-R
fprintf('\n=== Ewaluacja P-R (IoU = %.1f) ===\n', EVAL_IOU);

AP_values = nan(numel(test_files), 1);

for img_idx = 1:numel(test_files)
    dets   = all_dets{img_idx};
    scores = all_scores{img_idx};
    gt     = gt_boxes{img_idx};

    if isempty(gt)
        fprintf('  Obraz %2d : brak GT – pomijam\n', img_idx); continue;
    end
    if isempty(dets)
        fprintf('  Obraz %2d : brak detekcji – AP = 0.000\n', img_idx);
        AP_values(img_idx) = 0; continue;
    end

    [P, R, sc] = calcpr(dets, scores, gt, EVAL_IOU);
    ap = trapz(R, P); 
    AP_values(img_idx) = ap;
    fprintf('  Obraz %2d : AP = %.3f  (detekcji=%d, GT=%d)\n', ...
        img_idx, ap, size(dets,1), size(gt,1));

    figure('Name', sprintf('P-R – obraz %d', img_idx), 'NumberTitle','off');
    plot(R, P, 'x-', 'LineWidth', 1.5);
    xlabel('Recall'); ylabel('Precision');
    title(sprintf('Krzywa P-R – obraz %d  |  AP = %.3f', img_idx, ap));
    grid on; xlim([0 1]); ylim([0 1]);
end

valid_AP = AP_values(~isnan(AP_values));
fprintf('\nmAP (średnia po %d obrazach): %.3f\n', numel(valid_AP), mean(valid_AP));

%% =========================================================
%  FUNKCJE POMOCNICZE
%% =========================================================

function files = collect_images(dir_path)
% Zbiera pliki obrazów wszystkich obsługiwanych rozszerzeń z katalogu.
    exts  = {'*.jpg','*.jpeg','*.png','*.bmp'};
    files = [];
    for e = exts
        files = [files; dir(fullfile(dir_path, e{1}))]; %#ok<AGROW>
    end
end

function files = collect_test_images(dir_path)
% Zbiera pliki testowe do ewaluacji i wizualizacji
    files = [];

    crop_ids = [1501, 1684, 1658, 1659];

    for idx = 1:4
        filename = sprintf('people_%d.jpg', idx);
        fullpath = fullfile(dir_path, filename);
        
        if isfile(fullpath)
            file_info = dir(fullpath);
            files = [files; file_info]; %#ok<AGROW>
        else
            warning('File not found: %s', fullpath);
        end
    end

    for idx2 = 1:numel(crop_ids)
        img_id = crop_ids(idx2);
        filename = sprintf('crop%06d.png', img_id);
        fullpath = fullfile(dir_path, filename);
        
        if isfile(fullpath)
            file_info = dir(fullpath);
            files = [files; file_info]; %#ok<AGROW>
        else
            warning('File not found: %s', fullpath);
        end
    end
end

% --------------------------------------------------------
function img = load_gray_resized(path, H, W)
% Wczytuje obraz, konwertuje do skali szarości i skaluje do [H×W].
    raw = imread(path);
    img = im2gray(raw);
    img = imresize(img, [H, W]);
end

% --------------------------------------------------------
function img = load_gray_full(path)
% Wczytuje obraz w skali szarości.
    raw = imread(path);
    img = im2gray(raw);
end

% --------------------------------------------------------
function boxes = parse_pascal_annotation(ann_path)
% Parsuje anotacje INRIA/PASCAL.
% Zwraca macierz Nx4 [x y w h] (format MATLAB bboxOverlapRatio).
    boxes = zeros(0, 4);
    fid = fopen(ann_path, 'r');
    if fid == -1
        return;
    end

    pattern = ['Bounding box for object \d+.*?:\s*' ...
               '\((\d+),\s*(\d+)\)\s*-\s*' ...
               '\((\d+),\s*(\d+)\)'];

    while ~feof(fid)
        line = fgetl(fid);

        if ~ischar(line)
            continue;
        end

        tok = regexp(line, pattern, 'tokens');

        if ~isempty(tok)
            nums = cellfun(@str2double, tok{1});

            x1 = nums(1);
            y1 = nums(2);
            x2 = nums(3);
            y2 = nums(4);

            w = x2 - x1;
            h = y2 - y1;

            boxes(end + 1, :) = [x1, y1, w, h];
        end
    end
    fclose(fid);
end

% --------------------------------------------------------
function map = load_gtruth_mat(mat_path, fallback_names)
    map = containers.Map('KeyType','char','ValueType','any');
    if ~isfile(mat_path), return; end

    prev = warning('off','all');
    try
        S = load(mat_path, 'gTruth');
    catch ME
        warning(prev);
        warning('load_gtruth_mat: %s', ME.message);
        return;
    end
    warning(prev);

    if ~isfield(S,'gTruth') || ~isa(S.gTruth,'groundTruth'), return; end

    gt  = S.gTruth;
    tbl = gt.LabelData;

    % Znajdź kolumnę Rectangle
    defs     = gt.LabelDefinitions;
    bbox_col = '';
    for c = 1:height(defs)
        if defs.Type(c) == labelType.Rectangle
            bbox_col = char(defs.Name(c)); break;
        end
    end
    if isempty(bbox_col)
        warning('load_gtruth_mat: brak etykiet Rectangle'); return;
    end

    n_rows = height(tbl);
    if n_rows == 0, return; end

    % Dopasuj wiersze do nazw plików
    if nargin >= 2 && numel(fallback_names) >= n_rows
        names = fallback_names(1:n_rows);
        fprintf('[GT] gTruth.mat: %d wpisów (nazwy z fallback_names)\n', n_rows);
    else
        warning('load_gtruth_mat: fallback_names za krótka lub brak (%d wierszy w tabeli)', n_rows);
        return;
    end

    for k = 1:n_rows
        [~, fname, ~] = fileparts(names{k});
        cell_val = tbl.(bbox_col){k};
        map(fname) = double(cell_val);
    end

    fprintf('[GT] Załadowano gTruth.mat: etykieta "%s", %d obrazów\n', bbox_col, n_rows);
end


% --------------------------------------------------------
function [dets_out, scores_out] = nms_filter(dets, scores, iou_thr)
% Non-Maximum Suppression: zachowuje okna z najwyższym wynikiem SVM,
% usuwa nakładające się okna (IoU > iou_thr).
    if isempty(dets)
        dets_out = dets;  scores_out = scores;  return;
    end
    dets_out   = zeros(0, 4);
    scores_out = zeros(1, 0);
    tmp_d = dets;
    tmp_s = scores;
    while ~isempty(tmp_d)
        [~, best]  = max(tmp_s);
        dets_out   = [dets_out;   tmp_d(best,:)];          %#ok<AGROW>
        scores_out = [scores_out, tmp_s(best)];            %#ok<AGROW>
        ratio = bboxOverlapRatio(tmp_d(best,:), tmp_d, 'Min');
        keep       = ratio' < iou_thr;
        keep(best) = false;
        tmp_d = tmp_d(keep, :);
        tmp_s = tmp_s(keep);
    end
end

% --------------------------------------------------------
function visualize_result(img, dets, scores, gt, img_idx, img_name)
% Wizualizuje ground truth i predykcje na jednym obrazie.
    figure('Name', sprintf('Detekcja – obraz %d', img_idx), ...
           'NumberTitle', 'off', 'Units', 'normalized', ...
           'Position', [0.05, 0.05, 0.9, 0.85]);

    ann = img;

    % Ground truth – żółte ramki
    if ~isempty(gt) && size(gt,1) > 0
        gt_lbls = repmat({'GT'}, size(gt,1), 1);
        ann = insertObjectAnnotation(ann, 'rectangle', gt, gt_lbls, ...
            'Color', 'yellow', 'TextBoxOpacity', 0.7, 'FontSize', 12);
    end

    % Predykcje – zielone ramki z wynikiem SVM
    if ~isempty(dets)
        pred_lbls = arrayfun(@(s) sprintf('SVM:%.2f',s), scores, ...
            'UniformOutput', false);
        ann = insertObjectAnnotation(ann, 'rectangle', dets, pred_lbls, ...
            'Color', 'green', 'TextBoxOpacity', 0.7, 'FontSize', 12);
    end

    imshow(ann);
    title(sprintf('[%d] %s  |  GT: %d   Detekcji: %d', ...
        img_idx, img_name, size(gt,1), size(dets,1)), ...
        'Interpreter', 'none');

    % Legenda
    legend_img = ones(20,1,3,'uint8') * 200;
    annotation('textbox',[0.01 0.01 0.3 0.05], ...
        'String','Żółty = Ground Truth  |  Zielony = Detekcja SVM', ...
        'FontSize', 9, 'EdgeColor','none', 'BackgroundColor','white');
end

% --------------------------------------------------------
function [P, R, sc] = calcpr(dets, scores, gt_rect, iou)
% Oblicza Precision i Recall – detekcje sortowane malejąco wg score,

    n_gt  = size(gt_rect, 1);

    [sorted_sc, sort_idx] = sort(scores(:)', 'descend'); 
    sorted_dets = dets(sort_idx, :);
    n_det = size(sorted_dets, 1);

    tp_vec     = zeros(n_det, 1);
    fp_vec     = zeros(n_det, 1);
    gt_matched = false(n_gt, 1);

    for d = 1:n_det
        if n_gt == 0
            fp_vec(d) = 1;
            continue;
        end

        ov = bboxOverlapRatio(gt_rect, sorted_dets(d,:));
        [max_ov, best_gt] = max(ov);

        if max_ov > iou && ~gt_matched(best_gt)
            tp_vec(d)          = 1;
            gt_matched(best_gt) = true; 
        else
            fp_vec(d) = 1;
        end
    end

    cum_tp = cumsum(tp_vec);
    cum_fp = cumsum(fp_vec);

    R_raw = cum_tp / n_gt;
    P_raw = cum_tp ./ (cum_tp + cum_fp);

    for k = n_det-1:-1:1
        P_raw(k) = max(P_raw(k), P_raw(k+1));
    end

    R  = [0;       R_raw];
    P  = [1;       P_raw];
    sc = [Inf,     sorted_sc];
end
