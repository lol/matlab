function [ w, b, prediction ] = ldam( sample, training, label )
%LDA K-Class Classifier (Multiclass) using K linear discriminant functions
%   Given 'data' is a N feature dataset.
%   w = weights. K * N.
%   b = biases. K * 1.
%   Each row of w alongwith the bias forms the discriminant function.
%
%   Prediction is made in accordance with Hastie, Tibshirani &
%   Friedman - "The Elements of Statistical Learning", 2nd ed., p. 109.

if ~isvector(label)
    error('Label must be a vector.')
end

if size(training, 1) ~= length(label)
    error('Number of data points and categorical labels should be equal.')
end

if size(training, 2) ~= size(sample, 2)
    error('Training and Testing Data should have same number of features.')
end

numTotalSamples = size(training, 1);
covInv = pinv(cov(training));
numClasses = length(unique(label));
classes = unique(label);

for i = 1:numClasses
    numClassMembers = length(find(ismember(label, classes(i))));
    
    meanClass(i, :) = mean(training(find(ismember(label, classes(i))), :), 1);
    
    w(i, :) = covInv * meanClass(i, :)';
    b(i) = -0.5 * meanClass(i, :) * covInv * meanClass(i, :)' + log(numClassMembers / numTotalSamples);
    
    d(:, i) = sample * w(i, :)' + b(i);
end

[larg, prediction] = max(d, [], 2);
prediction = classes(prediction);
end