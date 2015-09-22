function outFile = gdf2mat(inFile, classLabel2Keep)
%
%Input: 
%
%inFile, name of the .gdf file without the extension. (Converted from
%OpenVibe)
%
%classLabel2Keep, a vector containing the class labels of the trials
%   to be kept (you can choose e.g., to study only two classes).
%   labels: 1 (left hand), 2 (right hand), 3 (foot), 4 (tongue)
%
%passBand, an optional structure defining a frequency band in which
%   filtering all the EEG signals this structure is such that:
%       passBand.low= low cut-off frequency (in Hz)
%       passBand.high= high cut-off frequency (in Hz)
%   by default no filtering is done
%
%NOTE: The gdf files are opened using Biosig and it needs to be in the path
%
%This file is supposed to be used from the Evaluations directory or any
%directory where you have .gdf files

%Channels {'C3';'C4';'Nz';'FC3';'FC4';'C5';'C1';'C2';'C6';'CP3';'CP4'};
%Channels {'1'; '2'; '3'; '4';   '5';  '6';'7'; '8'; '9'; '10';  '11'};

%reading the gdf file

outFile = strcat(inFile, '.mat');
inFile = strcat(inFile, '.gdf');

[s, h] = sload(inFile,0,'OVERFLOWDETECTION:OFF');
EEGdata.s = h.SampleRate;
fs = EEGdata.s;				%SamplingRate 
nbChannels = h.NS;
s = s(:,1:nbChannels); %we remove few channels if required


segmentLength = 2;      %we use a two seconds time window
segmentOffset = 2.5;    %since t=1 our time window starts 2.5s after start of trial.
stimCodes = [769 770];

%keeping only the trials from the required classes, if needed
if ~exist('classLabel2Keep','var')
    classLabel2Keep = [1 2 3 4];
end

stimCodes = stimCodes(classLabel2Keep);
%channelList = {'C3';'C4';'Nz';'FC3';'FC4';'C5';'C1';'C2';'C6';'CP3';'CP4'};
%channelList = {'C3';'Cz';'C4';'FCz';'CPz'};


% Defining the parameter of a Butterworth filter
passBand.low = 8;
passBand.high = 30;
order = 5; 
lowFreq = passBand.low * (2/fs);
highFreq = passBand.high * (2/fs);
[B A] = butter(order, [lowFreq highFreq]);

%if required, band-pass filter the signal in a given frequency band using a butterworth filter of order 5
%if exist('passBand','var')
    disp('band-pass filtering');
    s = filter(B,A,s);
%end

%counting the total number of trials for the kept classes
        
nbTrials = sum(ismember(h.EVENT.TYP,stimCodes));
EEGdata.y = zeros(1,nbTrials);
        
disp(['nbTrials: ' num2str(nbTrials)]);        
EEGdata.x = zeros((segmentLength * fs)+1, nbChannels, nbTrials);

%extracting the two second long segments for each trial
currentTrial = 1;
allTrialCount = 1;
for e = 1:length(h.EVENT.TYP)
    code = h.EVENT.TYP(e);
        if ismember(code,stimCodes)                    
            EEGdata.y(currentTrial) = code - 768;
            pos = h.EVENT.POS(e);                    
            range = pos+((fs*segmentOffset):(fs*(segmentOffset+segmentLength)));                    
            EEGdata.x(:,:,currentTrial) = s(range,:);
            currentTrial = currentTrial + 1;
        end
end

EEGdata.c = h.Label;

EEGSignals = EEGdata;

%EEGSignals = cell(1,1);
%subjectNo = 1
%EEGSignals{subjectNo} = EEGdata;
               
%% Saving the Results to the appropriate Matlab files
save(outFile, 'EEGSignals');