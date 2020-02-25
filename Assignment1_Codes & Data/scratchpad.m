%% 1. Features, Classes, and Linear regression
%% Load data
mTrain = load('data\mTrainData.txt');

%% Split data
Xtr = mTrain(:, 1);
Ytr = mTrain(:, 2);

%% Plot All Data
figure;
plot(Xtr, Ytr,'bo');
title('Plotting all training data points');

%% Plot Some Data
figure;
plot(Xtr(1:20), Ytr(1:20), 'bo');
title('Plotting 20 first training data points');

%% Increasing complexity
Xtr_2 = [ones(size(Xtr,1) ,1), Xtr , Xtr.^2];
learner = linearReg(Xtr_2 ,Ytr);	% train a linear regression learner
yhat = predict(learner , Xtr_2);	% use it for prediction

%% Using linearReg object
xline = [0:.01:2]'; % transpose : make a column vector , like training x
yline = predict( learner, polyx (xline, 2) ); % assuming quadratic features
figure;
plot(xline, yline);