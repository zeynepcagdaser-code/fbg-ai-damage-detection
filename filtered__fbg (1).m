

clc;
clear;
close all;

% Veri setini oku
data = readtable('fbg_simulink_labeled_dataset.csv');

% Gürültülü sinyali al
signal_noisy = data.delta_lambda_noisy;

% Zaman bilgisi
time = data.time;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILTRELEME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hareketli ortalama filtresi
windowSize = 15;

signal_filtered = movmean(signal_noisy, windowSize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YENI TABLO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data.delta_lambda_filtered = signal_filtered;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CSV KAYDET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writetable(data,'fbg_filtered_dataset.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GRAFIK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;

plot(time, signal_noisy, '.', 'DisplayName', 'Noisy Signal');
hold on;

plot(time, signal_filtered, 'LineWidth', 2, ...
    'DisplayName', 'Filtered Signal');

xlabel('Time');
ylabel('Delta Lambda (nm)');
title('FBG Signal Filtering');

legend;
grid on;

disp('Filtrelenmis dataset olusturuldu');

