clc;
clear all;

freqBands = [10, 15, 12];
%[s, h] = sload('ssvep-training-arjun-[2016.02.11-14.35.48].gdf', 0, 'OVERFLOWDETECTION:OFF');
%[s, h] = sload('ssvep-training-shiva-[2016.01.31-20.34.25].gdf', 0, 'OVERFLOWDETECTION:OFF');
[s, h] = sload('ssvep-record-train-[2016.04.09-10.38.44].gdf', 0, 'OVERFLOWDETECTION:OFF');
%[s, h] = sload('ssvep-record-train-indra-3-[2016.03.31-23.42.46].gdf', 0, 'OVERFLOWDETECTION:OFF');
%[s, h] = sload('ssvep-training-samit-[2016.02.09-15.55.56].gdf', 0, 'OVERFLOWDETECTION:OFF');
%[s, h] = sload('ssvep-record-train-prithvi-1-[2016.04.01-13.16.54].gdf', 0, 'OVERFLOWDETECTION:OFF');
fs = h.SampleRate;
numChannels = h.NS;
s = s(:, 1:numChannels); % selection of channels
%numChannels = 4;
%s = s(:, [2, 3, 5, 6]);

stimCodes = [33024, 33025, 33026, 33027];
numClasses = size(stimCodes, 2) - 1;

% Samples considered for training. From 1.000 to 7.999 sec.
flickerStart = 1;   % default = 1
flickerEnd = 8;     % could also be called last offset
samplesTrain = (flickerEnd - flickerStart) * fs; % 1750 samples @ fs = 250 Hz
startOffset = flickerStart * fs;

% stimCoordinate is a matrix with each column representing coordinate in
% h.EVENT.TYP
% h.EVENT.POS points to the sample number in signal vector 's'

for i = 1:size(stimCodes, 2)
    stimCoordinate(:, i) = find(ismember(h.EVENT.TYP, stimCodes(i)));
end

stimCoordinate_flat = find(ismember(h.EVENT.TYP, stimCodes));

classSignal = [];
nonclassSignal = [];

epochTime = 0.5;            % in seconds
epochOverlap = 0.1;         % in seconds
overlap_factor = (epochTime - epochOverlap) / epochTime;
discardBuffer = (samplesTrain - (epochTime * fs)) / (epochOverlap * fs);

data = [];
label = [];

% Band-pass filtering is the first thing which happens and it is done over
% the complete signal and on all channels.
for i = 1:numClasses                 % 3 classes
    signal(:, :, i) = s;                        % array of matrices
    
    for j = 1:numChannels
        order = 4;
        % Band-pass filtering from -0.25 to +0.25 Hz
        lowFreq = (freqBands(i) - 0.25) * (2/fs);
        highFreq = (freqBands(i) + 0.25) * (2/fs);

        [B, A] = butter(order, [lowFreq, highFreq]);
        signal(:, j, i) = filter(B, A, signal(:, j, i));
    end
    
    tempData = [];
    for j = 1:size(stimCoordinate_flat, 1)
        feature = [];
        if ~ismember(stimCoordinate_flat(j), stimCoordinate(:, i+1))
            % trial sample chunk
            signalTrial = signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i);
            tempInner = [];
            % epoching
            for k = 1:numChannels
                timeEpoch = buffer(signalTrial(:, k), epochTime * fs, ceil(overlap_factor * epochTime * fs));
                timeEpoch = timeEpoch(:, size(timeEpoch, 2) - discardBuffer:end);
                tempInner = [tempInner; log(1 + mean(timeEpoch .^ 2))];
            end
            feature = [feature; tempInner'];
            
            labelChunk = size(feature, 1);
            label((j - 1) * labelChunk + 1:j * labelChunk, i) = 4;
            % Using number 4 for non-class features.
        else
            % trial sample chunk
            signalTrial = signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i);
            tempInner = [];
            % epoching
            for k = 1:numChannels
                timeEpoch = buffer(signalTrial(:, k), epochTime * fs, ceil(overlap_factor * epochTime * fs));
                timeEpoch = timeEpoch(:, size(timeEpoch, 2) - discardBuffer:end);
                tempInner = [tempInner; log(1 + mean(timeEpoch .^ 2))];
            end
            feature = [feature; tempInner'];
            
            labelChunk = size(feature, 1);
            label((j - 1) * labelChunk + 1: j * labelChunk, i) = stimCodes(i + 1) - stimCodes(1);
        end
        tempData = [tempData; feature];
    end
    data(:, :, i) = tempData;
end

idealDecision = sum(~ismember(label, 4) .* label, 2);
labelFlat = idealDecision;
labelFlat(labelFlat == 0) = 4;

% data is 2112 x 6 x 3 for 1 to 8 second duration.
% label is 2112 x 3

concatData = [];
concatLabel = [];
for i = 1:numClasses + 1
    rowNo = find(ismember(labelFlat, i));
    tempConcat = [];
    for j = 1:numClasses
        tempConcat = [tempConcat, data(rowNo, :, j)];
    end
    %concatData(:, :, i) = tempConcat;
    concatData = [concatData; tempConcat];
end


concatLabel = [];
labelChunk = size(data, 1) / (numClasses + 1);
for i = 1:numClasses + 1
    concatLabel(size(concatLabel, 2) + 1 : size(concatLabel, 2) + labelChunk) = i;
end    

order = [];

%fprintf('=*=*=*=*=*=*=*=*= Class %d =*=*=*=*=*=*=*=*=\n', i);
fprintf('\n--- Resubstitution ---\n');

model = svmtrain(concatLabel', concatData, '-s 0 -t 2 -q');
predict = svmpredict(concatLabel', concatData, model, '-q');
confMat = confusionmat(concatLabel, predict)
perc = bsxfun(@rdivide, confMat, sum(confMat, 2)) * 100

fprintf('\n--- KFold Crossvalidation ---\n');
confMatK = zeros(size(confMat));
indices = crossvalind('Kfold', label(:, 1), 10);    %10 fold
for j = 1:10
    test = (indices == j);
    train = ~test;
    kmodel = svmtrain(concatLabel(train)', concatData(train, :), '-s 0 -t 2 -q');
    kpredict = svmpredict(concatLabel(test)', concatData(test, :), kmodel, '-q');
    confMatKj = confusionmat(concatLabel(test), kpredict);
    accK(j) = 100 * sum(diag(confMatKj)) / sum(sum(confMatKj));
    confMatK = confMatK + confMatKj;
end
confMatK
percK = bsxfun(@rdivide, confMatK, sum(confMatK, 2)) * 100