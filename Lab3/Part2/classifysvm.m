%% Parametry działania
% Powtarzalne wyniki
close all ;
rng('default') ;

% Liczba obrazów treningowych na klasę
cnt_train = 70 ;

% Liczba obrazów testowych na klasę
cnt_test = 30;

% Wybrane klasy obiektów
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Liczba cech wybierana na każdym obrazie
feats_det = 100;

% Metoda wyboru cech (true - jednorodnie w całym obrazie, false - najsilniejsze)
feats_uniform = true;

% Wielkość słownika
words_cnt = 30 ;

% Detekcja cech
% Ładowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

imds_full = imageDatastore("indoor/indoorCVPR_09/Images/", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds_full)

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
[imds, imtest] = splitEachLabel(imds_full, cnt_train, cnt_test, 'Include', img_classes);
%countEachLabel(imds)

% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

% Przygotowanie listy przechowującej indeksy plików i punktów charakterystycznych
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

% Obliczenie deskryptorów punktów charakterystycznych
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

% Tworzenie słownika

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Wizualizacja wyliczonych słów

% Wyznaczenie histogramów słów dla każdego obrazu treningowego
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

%% SVM - przykład
% Uczenie wieloklasowego klasyfikatora SVM o parametrach C i gamma.
% Rozpoznawanie wielu klas opiera się na regule one-vs-one
C = 0.1 ;
gamma = 0.1 ;
temp = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', C, 'KernelScale', gamma) ;
model = fitcecoc(file_hist, imds.Labels, 'Learners', temp) ;
train_err = loss(model, file_hist, imds.Labels, 'Lossfun', 'classiferror') ;
test_err = loss(model, test_hist, imtest.Labels, 'Lossfun', 'classiferror') ;
fprintf(1,'train_acc: %f, test_acc: %f\n', 1-train_err, 1-test_err) ;

% Kroswalidacja klasyfikatora w podziale na zbioru 4:1
modelcv = crossval(model, 'KFold', 5) ; % Model 'kroswalidowany'
modelcv.Trained % Model 'kroswalidowany' zawiera w sobie faktycznie 5 modeli - każdy uczony przy innym podziale zbioru
cv_err = kfoldLoss(modelcv) ; % Zagregowany błąd kroswalidacji
fprintf(1,'cross_validation_accuracy: %f\n', 1-cv_err) ;


%% Funkcje pomocnicze

function pts = getFeaturePoints(I, pts_det, pts_uniform)
    if size(I, 3) > 1
        I2 = rgb2gray(I);
    else
        I2 = I;
    end
    
    pts = detectSURFFeatures(I2, 'MetricThreshold', 100);
    if pts_uniform
        pts = selectUniform(pts, pts_det, size(I));
    else
        pts = pts.selectStrongest(pts_det);
    end
end

function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end