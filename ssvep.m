freq = [10, 15, 12];
[s, h] = sload('ssvep-training-shiva-[2016.01.31-20.34.25].gdf', 0, 'OVERFLOWDETECTION:OFF');
fs = h.SampleRate;
numChannels = h.NS;
s = s(:, 1:numChannels); % selection of channels

stimCodes = [33024, 33025, 33026, 33027];
numClasses = size(stimCodes,2) - 1;

% Samples considered for training. From 1.000 to 7.999 sec
flickerStart = 1;
flickerEnd = 8;     % could also be called last offset
samplesTrain = (flickerEnd - flickerStart) * fs; % 1750 samples @ fs = 250 Hz
startOffset = flickerStart * fs;

% stimCoordinate is a matrix with each colum representing coordinate in
% h.EVENT.TYP
% h.EVENT.POS points to the sample number in signal vector 's'

for i = 1:size(stimCodes,2)
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
        passBand.low = freq(i) - 0.25;          % filtering each channel
        passBand.high = freq(i) + 0.25;
        order = 4;
        lowFreq = passBand.low * (2/fs);
        highFreq = passBand.high * (2/fs);

        [B, A] = butter(order, [lowFreq highFreq]);
        signal(:, j, i) = filter(B, A, signal(:, j, i));
    end
    
    %for j = 1:size(stimCoordinate,1)
    %    classSignal = [classSignal; signal(h.EVENT.POS(stimCoordinate(j, i+1)):h.EVENT.POS(stimCoordinate(j, i+1)) + samplesTrain - 1, :, i)];
    %end
    
    for j = 1:size(stimCoordinate_flat,1)
        if ~ismember(stimCoordinate_flat(j), stimCoordinate(:, i+1))
            nonclassSignal = [nonclassSignal; signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i)];
        else
            classSignal = [classSignal; signal(h.EVENT.POS(stimCoordinate_flat(j)) + startOffset:h.EVENT.POS(stimCoordinate_flat(j)) + startOffset + samplesTrain - 1, :, i)];
        end
    end
end

% classSignal is a 42000 * 6 matrix. Three 14000 * 6 matrices are
% concatenated one after the other.
% nonClass signal is a 12600 * 6 matrix. Three 42000 * 6 matrices are
% concatenated one after the other.


% Just for verification
%signal(h.EVENT.POS(stimCoordinate(2, 2)) + startOffset,:,1)
%classSignal(1751,:)
%signal(h.EVENT.POS(stimCoordinate(1, 3)) + startOffset,:,2)
%classSignal(14001,:)


%classSignal1 = classSignal(1:14000, :);
%classSignal2 = classSignal(14001:28000, :);
%classSignal3 = classSignal(28001:42000, :);

%clearvars classsignal
%x = [];
%for j = 1:size(stimCoordinate_flat,1)
%   if ~ismember(stimCoordinate_flat(j), stimCoordinate(:, i+1))
        %x = [x; stimCoordinate_flat(j)];
    %end
%end

epochTime = 0.5;            % in seconds
epochOverlap = 0.1;         % in seconds
overlap_factor = (epochTime - epochOverlap) / epochTime;

discardBuffer = (samplesTrain - (epochTime * fs)) / (epochOverlap * fs);        % buffer introduces some zero padding in the beginning

% mat = buffer(classSignal(1:1750,1), epochTime * fs, ceil(overlap_factor * epochTime * fs));

feature = [];

for k = 1:size(classSignal,1)/numClasses:size(classSignal,1)
    for i = 1:size(stimCoordinate,1)
       tempInner = [];
       for j = 1:numChannels
            mat = buffer(classSignal((i - 1) * samplesTrain + 1 + (k-1):i * samplesTrain + (k-1), j), epochTime * fs, ceil(overlap_factor * epochTime * fs));
            mat = mat(:, size(mat,2) - discardBuffer:end);              % buffer introduces some zero padding in the beginning
            tempInner = [tempInner; log(1 + mean(mat .^ 2))];
       end
       feature = [feature; tempInner'];
    end
end

unfeature = [];

for k = 1:size(classSignal,1)/numClasses:size(classSignal,1)
    for i = 1:numClasses * size(stimCoordinate,1)
       tempInner = [];
       for j = 1:numChannels
            mat = buffer(nonclassSignal((i - 1) * samplesTrain + 1 + (k-1):i * samplesTrain + (k-1), j), epochTime * fs, ceil(overlap_factor * epochTime * fs));
            mat = mat(:, size(mat,2) - discardBuffer:end);              % buffer introduces some zero padding in the beginning
            tempInner = [tempInner; log(1 + mean(mat .^ 2))];
       end
       unfeature = [unfeature; tempInner'];
    end
end

% Divide feature and unfeature into 3 sets of class and non-class feature
% vectors to train classifier.

% for verification
% x = [];
% for i = 1:size(stimCoordinate,1)
%    for j = 1:numChannels
%        x = [x; (i - 1) * samplesTrain + 1; i * samplesTrain];
%    end
%end
