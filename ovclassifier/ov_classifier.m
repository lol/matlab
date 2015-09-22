function ov_classifier(trainFile, testFile)
%
% 
% Usage: ov_classifier(trainFile, testFile)
%
% Filenames should be without the .mat extension
%
%

trainFile = strcat(trainFile, '.mat');
testFile = strcat(testFile, '.mat');

%% Loading the Mat files from the created training and testing set 
train = load(trainFile);
test = load(testFile);

%% Extracting CSP Features from each trial
st = 1;
nbFilterPairs = 1;

CSPMatrix = ov_learnCSP(train.EEGSignals);
trainFeatures = ov_extractCSPFeatures(train.EEGSignals, CSPMatrix, nbFilterPairs);  
testFeatures = ov_extractCSPFeatures(test.EEGSignals, CSPMatrix, nbFilterPairs);

%% LDA Classifier II
%Re-Substitution
[ldaClass, ldaResubErr] = classify(trainFeatures(:,1:end-1),trainFeatures(:,1:end-1),trainFeatures(:,end)); %LDA Classifier o/p
disp('Classifier Accuracy-Training Data');
ldaResubAccuracy = (1 - ldaResubErr)*100                       %Calculating Classifier Accuracy
disp('Displaying Confusion Matrix for training data');
ldaResubCM = confusionmat(trainFeatures(:,end),ldaClass)

%% k-Fold Cross-Validation

ldaClassFun = @(xtrain,ytrain,xtest)(classify(xtest,xtrain,ytrain));
disp('LDA k-Fold Cross-Validation Accuracy');
%ldaCVErr  = crossval('mcr',trainFeatures(:,1:end-1),trainFeatures(:,end),'predfun',ldaClassFun,'kfold',5);
ldaCVErr = crossval('mcr',trainFeatures(:,1:end-1), trainFeatures(:,end), 'predfun', ldaClassFun, 'leaveout', 1);
ldaCVaccuracy = (1 - ldaCVErr)*100
         
%% Testing Classifier
%Testing Data
[ldaClass] = classify(testFeatures(:,1:end-1), trainFeatures(:,1:end-1), trainFeatures(:,end)); %LDA Classifier o/p
disp('Classifier Accuracy-Testing Data');
disp('Displaying Confusion Matrix for testing data');
ldaTestCM = confusionmat(testFeatures(:,end), ldaClass)
bad = (ldaClass ~= testFeatures(:,end));   
ldaTestaccuracy = (1 - (sum(bad)/size(ldaClass,1)))*100