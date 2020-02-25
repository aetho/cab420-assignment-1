%% 2. kNN Regression
%% (a) Using the knnRegress class, implement (add code to) the predict
...function to make it functional.
	
% predict has been implemented in predict.m of kNNRegress

%% Setup
clear; clc; close all;

% Load training data
mTrain = load('data\mTrainData.txt');

% Split training data into features and labels (Xtr and Ytr)
Xtr = mTrain(:, 1);
Ytr = mTrain(:, 2);

% Load test data
mTest = load('data\mTestData.txt');

% Split data test data into features and labels (Xte and Yte)
Xte = mTest(:, 1);
Yte = mTest(:, 2);
%% (b) Plotting kNN regression for k: 1,2,3,5,10,50

k = [1 2 3 5 10 50];
nnLearners(size(k)) = knnRegress;
xline = 0:0.01:1;
yhat = zeros(length(k),length(xline));
for i = 1:length(k)
	nnLearners(i) = knnRegress(k(i), Xtr, Ytr);
	yhat(i,:) = predict(nnLearners(i), xline');
end

figure
% Plot data
plot(Xtr, Ytr,'bo');

% Plot learners
hold on
plot(xline, yhat);
hold off

xlabel('x');
ylabel('y');
title('Training Data And kNN Regression Learners');
legend('Training data', 'k = 1', 'k = 2', 'k = 3', 'k = 5', 'k = 10', 'k = 50', 'Location', 'northeastoutside');

% How does the choice of k relate to the “complexity” of the regression function?
%	From the plot it can be seen that as k increases, the regression
%	function's complexity decreases.


%% (c) What kind of functions can be output by a nearest neighbor regression function?
%	The functions outputted by the nearest neighbor regression function are
%	all piece wise linear as all x values that share the same neighbors 
%	will correspond to the same y value meaning it is piece wise and linear.
%




