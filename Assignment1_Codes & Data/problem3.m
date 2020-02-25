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

%% (a) Plot both train and test MSE versus k on a log-log scale

% First 20 training data examples
Xtr20 = Xtr(1:20);
Ytr20 = Ytr(1:20);

k = 1:140;
nnLearners(size(k)) = knnRegress;		% holds knnRegress learners
yhatTr = zeros(length(k),length(Xtr));	% holds predictions on training data
yhatTe = zeros(length(k),length(Xte));	% holds predictions on testing data
mseTr = zeros(size(k));					% holds MSE of training data
mseTe = zeros(size(k));					% hold MSE of testing data

for i = 1:length(k)
	% Training learner with k=i and first 20 training examples
	nnLearners(i) = knnRegress(k(i), Xtr20, Ytr20);
	
	% Using model to predict training x values
	yhatTr(i,:) = predict(nnLearners(i), Xtr);
	
	% Using model to predict testing x values
	yhatTe(i,:) = predict(nnLearners(i), Xte);
	
	% Calculating MSE of training and test data
	mseTr(i) = immse(yhatTr(i,:)', Ytr);
	mseTe(i) = immse(yhatTe(i,:)', Yte);
end

% Storing MSE's for later use in (c)
mseArrayTr = mseTr;
mseArrayTe = mseTe;

figure;
hold on
loglog(k, mseTr, '-ro', 'MarkerSize', 2);
loglog(k, mseTe, '-go', 'MarkerSize', 2);
hold off

% 'hold on' breaks 'loglog' so have to manually set the axis to log again 
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

title('MSE vs k for 20 data points');
xlabel('k');
ylabel('MSE');
legend('Train', 'Test', 'Location', 'northwest');
axis([0 140 0 2]);

% GRAPH OBSERVATIONS
% The following was observed from the figure:
%	- Initially MSE decreases
%	- As k approaches the number of traning samples, MSE increases.
%	- When k >= number of training samples, MSE is constant

%% (b) Repeat (a), but use all the training data.

k = 1:140;
nnLearners(size(k)) = knnRegress;		% holds knnRegress learners
yhatTr = zeros(length(k),length(Xtr));	% holds predictions on training data
yhatTe = zeros(length(k),length(Xte));	% holds predictions on testing data
mseTr = zeros(size(k));					% holds MSE of training data
mseTe = zeros(size(k));					% hold MSE of testing data

for i = 1:length(k)
	% Training learner with k=i and all training examples
	nnLearners(i) = knnRegress(k(i), Xtr, Ytr);
	
	% Using model to predict training x values
	yhatTr(i,:) = predict(nnLearners(i), Xtr);
	
	% Using model to predict testing x values
	yhatTe(i,:) = predict(nnLearners(i), Xte);
	
	% Calculating MSE of training and test data
	mseTr(i) = immse(yhatTr(i,:)', Ytr);
	mseTe(i) = immse(yhatTe(i,:)', Yte);
end

% Storing MSE's for later use in (c)
mseArrayTr = [mseArrayTr; mseTr];
mseArrayTe = [mseArrayTe; mseTe];

figure;
hold on
loglog(k, mseTr, '-ro', 'MarkerSize', 2);
loglog(k, mseTe, '-go', 'MarkerSize', 2);
hold off

% 'hold on' breaks 'loglog' so have to manually set the axis to log again 
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

title('MSE vs k for all data points');
xlabel('k');
ylabel('MSE');
legend('Train', 'Test', 'Location', 'northwest');
axis([0 140 0 2]);


% GRAPH OBSERVATIONS
% The following differences between problem 1e and 3b were observed:
%	- In problem 1e, as the order of the model increased the complexity and
%	MSE of the model also increased. However, in problem 3b, as k increases
%	the complexity of the model decreases yielding a 'smoother' function.

%% (c) 4-fold cross-validation
nCV = 1:4;			% cross-validatons
kValues = 1:140;	% k values

nnLearners(size(nCV)) = knnRegress;	% Allocating space for learners

mseTr = zeros(length(kValues), length(nCV));	% All MSE of training data
mseTe = zeros(length(kValues), length(nCV));	% All MSE of testing data

for k = kValues
	for i = nCV
		idxS = (i-1)*20 + 1;			% Starting index of testing data
		idxE = idxS + 19;				% Ending index of testing data
		iTest = idxS:idxE;				% Indices for testing data
		iTrain = setdiff(1:140, iTest);	% Indices for training data

		nnLearners(i) = knnRegress(k, Xtr(iTrain), Ytr(iTrain));
		
		% Validating learner
		yHatTr = predict(nnLearners(i), Xtr(iTrain));
		mseTr(k,i) = immse(yHatTr, Ytr(iTrain));
		
		yHatTe = predict(nnLearners(i), Xtr(iTest));
		mseTe(k,i) = immse(yHatTe, Ytr(iTest));
	end
end

% Averaging MSE's
mseAvgTr = mean(mseTr, 2);
mseAvgTe = mean(mseTe, 2);

% Storing MSE's for later use in (c)
mseArrayTr = [mseArrayTr; mseAvgTr'];
mseArrayTe = [mseArrayTe; mseAvgTe'];

figure;
hold on
loglog(kValues, mseAvgTr, '-ro', 'MarkerSize', 2);
loglog(kValues, mseAvgTe, '-go', 'MarkerSize', 2);
hold off

% 'hold on' breaks 'loglog' so have to manually set the axis to log again 
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

title('4-Fold CV MSE vs k for all data points');
xlabel('k');
ylabel('MSE');
legend('Train', 'Test', 'Location', 'northwest');
axis([0 140 0 2]);

% Plotting (a), (b), and (c)
figure;
hold on
for i = 1:3
	loglog(kValues, mseArrayTr(i,:), '-o', 'MarkerSize', 2);
	loglog(kValues, mseArrayTe(i,:), '-o', 'MarkerSize', 2);
end
hold off

% 'hold on' breaks 'loglog' so have to manually set the axis to log again 
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

title('MSE vs k for all data points');
xlabel('k');
ylabel('MSE');
legend('(a) Train', '(a) Test', '(b) Train', '(b) Test',...
	'(c) Train', '(c) Test', 'Location', 'northwest');
axis([0 140 0 2]);

% GRAPH OBSERVATION
% 4-fold cross-validation yields similar accuracy to (b) while using less 
% data. In part (b) 200 data samples (140 for training and 60 for testing)
% were used while 4-fold cross-validation use only 140. Thus, in cases
% where large amounts of data is not available, hold-out and
% cross-validation can be used to achieve accuracy with less data.








