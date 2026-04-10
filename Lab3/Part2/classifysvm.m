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

file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    if size(I, 3) > 1
        Ig = rgb2gray(I);
    else
        Ig = I;
    end
    curr_features = extractFeatures(Ig, all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);

file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    if size(I, 3) > 1
        Ig = rgb2gray(I);
    else
        Ig = I;
    end
    feats = extractFeatures(Ig, pts);
    test_hist(i,:) = wordHist(feats, words);
end

%% Punkt 1 - Uruchomienie SVM z domyślnymi parametrami (przykład ze skryptu)
% Demonstracja działania klasyfikatora przed optymalizacją
close all;

C_default = 0.1;
gamma_default = 0.1;

temp_default = templateSVM('KernelFunction', 'gaussian', ...
    'BoxConstraint', C_default, 'KernelScale', gamma_default);
model_default = fitcecoc(file_hist, imds.Labels, 'Learners', temp_default);

train_err_default = loss(model_default, file_hist,    imds.Labels,  'Lossfun', 'classiferror');
test_err_default = loss(model_default, test_hist, imtest.Labels, 'Lossfun', 'classiferror');

fprintf('\n--- Wyniki SVM z domyślnymi parametrami (C=%.2f, gamma=%.2f) ---\n', ...
    C_default, gamma_default);
fprintf('Accuracy treningowa:  %.4f (%.2f%%)\n', 1-train_err_default, (1-train_err_default)*100);
fprintf('Accuracy testowa:     %.4f (%.2f%%)\n', 1-test_err_default,  (1-test_err_default)*100);

modelcv_default = crossval(model_default, 'KFold', 5);
cv_err_default  = kfoldLoss(modelcv_default);
fprintf('CV accuracy (k=5):    %.4f (%.2f%%)\n', 1-cv_err_default, (1-cv_err_default)*100);

%% Punkt 2 - Grid search z k-fold cross-validation
% Wyczerpujące przeszukiwanie przestrzeni hiperparametrów C i gamma.
% Uzasadnienie k=5: przy 210 obrazach treningowych każdy fold zawiera ~42
% obrazy walidacyjne (14 na klasę) - wystarczająca liczba dla wiarygodnej
% oceny. Większe k (np. 10) zwiększyłoby dokładność oceny, ale wydłużyło
% czas grid searcha proporcjonalnie.
close all;

% Siatka wartości - skala logarytmiczna
% Zmodyfikuj zakresy jeśli optimum wypada na granicy siatki
C_values = logspace(-2, 3, 8);   % 0.01 ... 1000
gamma_values = logspace(-3, 2, 8);   % 0.001 ... 100
k_folds = 5;

nC = length(C_values);
nG = length(gamma_values);

% Macierze wyników grid searcha
acc_grid_cv = zeros(nC, nG);   % średnia accuracy CV
std_grid_cv = zeros(nC, nG);   % odchylenie std accuracy CV

fprintf('\n--- Grid Search %dx%d = %d kombinacji, k=%d ---\n', ...
    nC, nG, nC*nG, k_folds);
fprintf('Szacowany czas: kilka-kilkanaście minut.\n\n');

total_iter = nC * nG;
iter = 0;

for i = 1:nC
    for j = 1:nG
        iter = iter + 1;
        fprintf('[%3d/%3d] C=%.4f, gamma=%.4f ... ', ...
            iter, total_iter, C_values(i), gamma_values(j));
        
        temp = templateSVM('KernelFunction', 'gaussian', ...
            'BoxConstraint',  C_values(i), ...
            'KernelScale',    gamma_values(j));
        
        model_cv = fitcecoc(file_hist, imds.Labels, ...
            'Learners', temp);
        
        % k-fold cross-validation na zbiorze treningowym
        cv_model = crossval(model_cv, 'KFold', k_folds);
        
        % kfoldLoss zwraca błąd dla każdego foldu osobno przy 'Mode','individual'
        fold_errors = kfoldLoss(cv_model, 'Mode', 'individual');
        fold_accs = (1 - fold_errors) * 100;
        
        acc_grid_cv(i, j) = mean(fold_accs);
        std_grid_cv(i, j) = std(fold_accs);
        
        fprintf('CV acc = %.2f%% (+/- %.2f%%)\n', ...
            acc_grid_cv(i,j), std_grid_cv(i,j));
    end
end

%% Punkt 2c - Raportowanie wyników grid searcha
close all;

fprintf('\n========== TABELA WYNIKÓW GRID SEARCH (Accuracy CV %%) ==========\n');
fprintf('%-12s', 'C \\ gamma');
for j = 1:nG
    fprintf('  %8.4f', gamma_values(j));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 12 + nG*10));
for i = 1:nC
    fprintf('%-12.4f', C_values(i));
    for j = 1:nG
        fprintf('  %8.2f', acc_grid_cv(i,j));
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('=', 1, 12 + nG*10));

% Najlepsza kombinacja
[best_acc_flat, best_idx] = max(acc_grid_cv(:));
[best_i, best_j] = ind2sub([nC, nG], best_idx);
best_C = C_values(best_i);
best_gamma = gamma_values(best_j);

fprintf('\nNajlepsza kombinacja z grid searcha:\n');
fprintf('  C     = %.6f\n', best_C);
fprintf('  gamma = %.6f\n', best_gamma);
fprintf('  CV accuracy = %.2f%% (+/- %.2f%%)\n', ...
    acc_grid_cv(best_i, best_j), std_grid_cv(best_i, best_j));

% --- Wizualizacja 3D przestrzeni parametrów ---
figure('Name', 'Grid Search - przestrzeń parametrów 3D');
[C_mesh, G_mesh] = meshgrid(log10(gamma_values), log10(C_values));
surf(C_mesh, G_mesh, acc_grid_cv, 'EdgeColor', 'k', 'FaceAlpha', 0.85);
colormap(parula);
colorbar;
xlabel('log_{10}(\gamma)');
ylabel('log_{10}(C)');
zlabel('Accuracy walidacyjna (%)');
title('Grid Search SVM - przestrzeń hiperparametrów');
hold on;
% Zaznacz optimum na wykresie
plot3(log10(best_gamma), log10(best_C), best_acc_flat + 0.5, ...
    'r*', 'MarkerSize', 14, 'LineWidth', 2);
legend('Accuracy CV', sprintf('Optimum (C=%.3f, \\gamma=%.4f)', best_C, best_gamma), ...
    'Location', 'best');
hold off;

% --- Mapa ciepła (heatmap) dla lepszej czytelności ---
figure('Name', 'Grid Search - mapa ciepła');
imagesc(log10(C_values), log10(gamma_values), acc_grid_cv');
colormap(parula);
colorbar;
xlabel('log_{10}(C)');
ylabel('log_{10}(\gamma)');
title('Grid Search SVM - mapa accuracy walidacyjnej (%)');
ax = gca;
ax.XTick = log10(C_values);
ax.XTickLabel = arrayfun(@(x) sprintf('%.2g', x), C_values, 'UniformOutput', false);
ax.YTick = log10(gamma_values);
ax.YTickLabel = arrayfun(@(x) sprintf('%.2g', x), gamma_values, 'UniformOutput', false);
% Nanieś wartości na mapę
for i = 1:nC
    for j = 1:nG
        text(log10(C_values(i)), log10(gamma_values(j)), ...
            sprintf('%.1f', acc_grid_cv(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'w');
    end
end

% !!!
input('Naciśnij Enter żeby kontynuować...');
%% Punkt 2b/2d - Finalny model z najlepszymi parametrami + wyniki testowe
close all;

fprintf('\n--- Trening finalnego modelu: C=%.6f, gamma=%.6f ---\n', ...
    best_C, best_gamma);

temp_best  = templateSVM('KernelFunction', 'gaussian', ...
    'BoxConstraint', best_C, 'KernelScale', best_gamma);
model_best = fitcecoc(file_hist, imds.Labels, 'Learners', temp_best);

% Wyniki na zbiorze treningowym
train_err_best = loss(model_best, file_hist, imds.Labels, 'Lossfun', 'classiferror');
train_acc_best = (1 - train_err_best) * 100;

% Wyniki na zbiorze testowym
test_err_best = loss(model_best, test_hist, imtest.Labels, 'Lossfun', 'classiferror');
test_acc_best = (1 - test_err_best) * 100;

fprintf('Accuracy treningowa:  %.2f%%\n', train_acc_best);
fprintf('Accuracy testowa:     %.2f%%\n', test_acc_best);

% Predykcje dla macierzy pomyłek i miar F1
preds_test = predict(model_best, test_hist);

% Macierz pomyłek
figure('Name', 'Macierz pomyłek - finalny model SVM');
cm = confusionchart(removecats(imtest.Labels), removecats(preds_test));
cm.Title = sprintf('Macierz pomyłek SVM (C=%.4f, gamma=%.4f)', best_C, best_gamma);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% --- Mikro i makro accuracy ---
classes = unique(imds.Labels);
n_classes = length(classes);
true_labels = imtest.Labels;

% Macierz pomyłek jako liczby
cm_matrix = confusionmat(removecats(true_labels), removecats(preds_test));

% Accuracy per klasa (do makro-uśrednienia)
per_class_acc = zeros(n_classes, 1);
for c = 1:n_classes
    TP = cm_matrix(c, c);
    FN = sum(cm_matrix(c, :)) - TP;
    FP = sum(cm_matrix(:, c)) - TP;
    TN = sum(cm_matrix(:)) - TP - FP - FN;
    per_class_acc(c) = (TP + TN) / (TP + FP + FN + TN);
end

% Precision, Recall, F1 per klasa
precision = zeros(n_classes, 1);
recall = zeros(n_classes, 1);
f1 = zeros(n_classes, 1);
for c = 1:n_classes
    TP = cm_matrix(c, c);
    FP = sum(cm_matrix(:, c)) - TP;
    FN = sum(cm_matrix(c, :)) - TP;
    precision(c) = TP / max(TP + FP, 1);
    recall(c) = TP / max(TP + FN, 1);
    f1(c) = 2 * precision(c) * recall(c) / max(precision(c) + recall(c), 1e-10);
end

% Mikro accuracy = łączna liczba poprawnych / wszystkie próbki
micro_acc = sum(diag(cm_matrix)) / sum(cm_matrix(:)) * 100;

% Makro accuracy = średnia accuracy po klasach
macro_acc = mean(per_class_acc) * 100;

% Makro F1 = średnia F1 po klasach
macro_f1 = mean(f1);

fprintf('\n========== RAPORT KLASYFIKACJI - FINALNY MODEL SVM ==========\n');
fprintf('%-15s  %10s  %10s  %10s\n', 'Klasa', 'Precision', 'Recall', 'F1');
fprintf('%s\n', repmat('-', 1, 50));
for c = 1:n_classes
    fprintf('%-15s  %10.4f  %10.4f  %10.4f\n', ...
        char(classes(c)), precision(c), recall(c), f1(c));
end
fprintf('%s\n', repmat('-', 1, 50));
fprintf('%-15s  %10s  %10s  %10.4f\n', 'Makro-avg', '-', '-', macro_f1);
fprintf('\nMikro-uśredniona accuracy:  %.2f%%\n', micro_acc);
fprintf('Makro-uśredniona accuracy:  %.2f%%\n', macro_acc);
fprintf('==============================================================\n');

%% Punkt 3 - Automatyczna optymalizacja (bayesopt) do porównania
% MATLAB oferuje wbudowaną optymalizację bayesowską jako alternatywę
% dla ręcznego grid searcha.
close all;

fprintf('\n--- Automatyczna optymalizacja (Bayesian Optimization) ---\n');
fprintf('Uwaga: może trwać kilka minut.\n');

opts_bayes = struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 40, ...
    'ShowPlots', true, ...
    'Verbose', 1);

model_auto = fitcecoc(file_hist, imds.Labels, ...
    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions', opts_bayes);

% Wyniki automatycznej optymalizacji
best_C_auto = model_auto.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
best_gamma_auto = model_auto.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;

test_err_auto = loss(model_auto, test_hist, imtest.Labels, 'Lossfun', 'classiferror');
test_acc_auto = (1 - test_err_auto) * 100;

fprintf('\n========== PORÓWNANIE METOD OPTYMALIZACJI ==========\n');
fprintf('%-25s  %10s  %10s  %12s\n', 'Metoda', 'C', 'gamma', 'Test acc (%)');
fprintf('%s\n', repmat('-', 1, 62));
fprintf('%-25s  %10.4f  %10.4f  %12.2f\n', ...
    'Domyślne parametry', C_default, gamma_default, (1-test_err_default)*100);
fprintf('%-25s  %10.4f  %10.4f  %12.2f\n', ...
    'Grid Search (ręczny)', best_C, best_gamma, test_acc_best);
fprintf('%-25s  %10.4f  %10.4f  %12.2f\n', ...
    'Bayesian Optimization', best_C_auto, best_gamma_auto, test_acc_auto);
fprintf('%s\n', repmat('=', 1, 62));

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