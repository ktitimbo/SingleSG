clear; close all; clc

addpath(fullfile(fileparts(pwd), 'MatlabFunctions'));

setGraphicsDefault();

[DKPath, FLPath] = getDKFLPaths(0.2);

bin = [1, 1];

numImagesPerGroup = 30;

%% Dark Images
DKSource = filterImgs(parseImgs(DKPath), 'Bin', bin);
DKImgs = zeros(2160 / bin(2), 2560 / bin(1), numel(DKSource));
for i = 1:numel(DKSource)
    DKImgs(:, :, i) = readB16(DKSource(i).Filename);
end
DKMean = mean(DKImgs, 3);
clear DKSource DKImgs i

save('img_dk.mat', 'DKMean', '-v7.3', '-nocompression')

%% Flat Images
FLSource = filterImgs(parseImgs(FLPath), 'Bin', bin);
FLImgs = zeros(2160 / bin(2), 2560 / bin(1), numel(FLSource));
for i = 1:numel(FLSource)
    FLImgs(:, :, i) = readB16(FLSource(i).Filename) - DKMean;
end
FLMean = mean(FLImgs, 3);
clear FLSource FLImgs i

save('img_fl.mat', 'FLMean', '-v7.3', '-nocompression')

%% Process Data
data = struct('Current_mA', {}, 'AmmeterRange_mA', {}, 'F1ProcessedImages', {}, 'F2ProcessedImages', {}, ...
    'PeakSeparationMean_um', {}, 'PeakSeparationStd_um', {});

ROI = 4 * [80 / bin(1), 80 / bin(2), 480 / bin(1), 420 / bin(2)];

ROIx = ROI(1):(ROI(1) + ROI(3) - 1);
ROIy = ROI(2):(ROI(2) + ROI(4) - 1);

p = 0.01;

x = linspace(ROIx(1), ROIx(end), 2^14);

fileinfo = appParse();

%% Save
for i = 1:numel(fileinfo)
    load(fileinfo(i).Filename);
    data(i).Current_mA = fileinfo(i).Current_mA;
    data(i).AmmeterRange_mA = fileinfo(i).Range_mA;
    BGMean = (mean(BG, 3) - DKMean) ./ FLMean;
    data(i).F1ProcessedImages = (double(F1) - DKMean) ./ FLMean - BGMean;
    data(i).F2ProcessedImages = (double(F2) - DKMean) ./ FLMean - BGMean;

    s = zeros(numImagesPerGroup, 1);

    for j = 1:numImagesPerGroup
        temp1 = data(i).F1ProcessedImages(:, :, j);
        temp2 = data(i).F2ProcessedImages(:, :, j);

        F11D = mean(temp1(ROIy, ROIx), 1);
        F21D = mean(temp2(ROIy, ROIx), 1);

        f1 = fit(ROIx', F11D', 'smoothingspline', 'SmoothingParam', p);
        f2 = fit(ROIx', F21D', 'smoothingspline', 'SmoothingParam', p);

        [~, idx1] = max(f1(x));
        [~, idx2] = max(f2(x));

        s(j) = abs(x(idx2) - x(idx1)) * bin(1) * 6.5; % in microns
        fprintf('     i=%d, j=%d.\n', i, j)
        clear temp1 temp2 F11D F21D f1 f2 idx1 idx2
    end

    data(i).PeakSeparationMean_um = mean(s);
    data(i).PeakSeparationStd_um = std(s);
    clear Background F1 F2
end

save('data000.mat', 'data', '-v7.3', '-nocompression')

%% Plot separation
figure; set(gcf, 'Position', [300, 200, 700, 550])
errorbar([data.Current_mA], [data.PeakSeparationMean_um], [data.PeakSeparationStd_um], ...
    [data.PeakSeparationStd_um], 0.015 * [data.AmmeterRange_mA],  0.015 * [data.AmmeterRange_mA], ...
    'LineWidth', 1.2)
xlabel('Coil Current (mA)')
ylabel('Peak Separation (um)')
title('Linear Plot')
grid on
exportgraphics(gca, 'Linear.pdf', 'ContentType', 'vector')

figure; set(gcf, 'Position', [350, 200, 700, 550])
errorbar([data.Current_mA], [data.PeakSeparationMean_um], [data.PeakSeparationStd_um], ...
    [data.PeakSeparationStd_um], 0.015 * [data.AmmeterRange_mA],  0.015 * [data.AmmeterRange_mA], ...
    'LineWidth', 1.2)
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('Coil Current (mA)')
ylabel('Peak Separation (um)')
title('Log Plot')
grid on
exportgraphics(gca, 'Log.pdf', 'ContentType', 'vector')