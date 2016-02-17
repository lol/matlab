clc;
clear all;

freqBands = [10, 15, 12];
[s, h] = sload('ssvep-training-samit-[2016.02.09-15.55.56].gdf', 0, 'OVERFLOWDETECTION:OFF');
fs = h.SampleRate;
numChannels = h.NS;
s = s(:, 1:numChannels); % selection of channels

stimCodes = [33024, 33025, 33026, 33027];
numClasses = size(stimCodes, 2) - 1;

% Samples considered for training. From 1.000 to 7.999 sec.
flickerStart = 1;
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
    
    for j = 1:size(stimCoordinate_flat, 1)
        if ~ismember(stimCoordinate_flat(j), stimCoordinate(:, i+1))
            nonclassSignal = [nonclassSignal; signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i)];
        else
            classSignal = [classSignal; signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i)];
        end
    end
end

% classSignal is a 42000 * 6 matrix. Three 14000 * 6 matrices are
% concatenated one after the other.
% nonClass signal is a 126000 * 6 matrix. Three 42000 * 6 matrices are
% concatenated one after the other.

epochTime = 0.5;            % in seconds
epochOverlap = 0.1;         % in seconds
overlap_factor = (epochTime - epochOverlap) / epochTime;

% buffer introduces some zero padding in the beginning, which needs to be
% discarded
discardBuffer = (samplesTrain - (epochTime * fs)) / (epochOverlap * fs);

% mat = buffer(classSignal(1:1750,1), epochTime * fs, ceil(overlap_factor * epochTime * fs));

feature = [];

for k = 1:size(classSignal, 1) / numClasses:size(classSignal, 1)
    for i = 1:size(stimCoordinate, 1)
       tempInner = [];
       for j = 1:numChannels
            timeEpoch = buffer(classSignal((i - 1) * samplesTrain + k:i * samplesTrain + k - 1, j), epochTime * fs, ceil(overlap_factor * epochTime * fs));
            timeEpoch = timeEpoch(:, size(timeEpoch, 2) - discardBuffer:end);
            tempInner = [tempInner; log(1 + mean(timeEpoch .^ 2))];
       end
       feature = [feature; tempInner'];
    end
end

unfeature = [];

for k = 1:size(classSignal, 1) / numClasses:size(classSignal, 1)
    for i = 1:numClasses * size(stimCoordinate, 1)
       tempInner = [];
       for j = 1:numChannels
            timeEpoch = buffer(nonclassSignal((i - 1) * samplesTrain + k:i * samplesTrain + k - 1, j), epochTime * fs, ceil(overlap_factor * epochTime * fs));
            timeEpoch = timeEpoch(:, size(timeEpoch, 2) - discardBuffer:end);
            tempInner = [tempInner; log(1 + mean(timeEpoch .^ 2))];
       end
       unfeature = [unfeature; tempInner'];
    end
end

% Divide feature and unfeature into 3 sets of class and non-class feature
% vectors to train classifier.
