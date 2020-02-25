%% 1. Features, Classes, and Linear Regression
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

%% (a) Plot the training data in a scatter plot
figure;
plot(Xtr, Ytr,'bo');
title('Training Data And Linear Regression Learners');
legend('Data', 'Location', 'northeastoutside');
xlabel('x');
ylabel('y');

%% (b) Create a linear regression learner using the above functions.
...Plot it on the same plot as the training data.

% Creating learner
learnerLinear = linearReg(polyx(Xtr,1), Ytr);

% Using learner to get predictions for 0:0.01:1
xline = 0:0.01:1;
yHatLinear = predict(learnerLinear, polyx(xline',1));

% Plotting
hold on
plot(xline, yHatLinear);
hold off
legend('Data', 'Linear learner');

%% (c) Create plots with the data and a higher-order polynomial (3, 5, 7, 9, 11, 13).

orders = 3:2:13;
learners(size(orders)) = linearReg;
xline = 0:0.01:1;
yhat = zeros(6,length(xline));
for i = 1:length(learners)
	learners(i) = linearReg(polyx(Xtr, orders(i)), Ytr);
	yhat(i,:) = predict(learners(i), polyx(xline',orders(i)));
end

hold on
plot(xline, yhat);
hold off

xlabel('x');
ylabel('y');
legend('Training data', 'Linear learner', '3rd-Order Learner',...
	'5th-Order Learner', '7th-Order Learner', '9th-Order Learner',...
	'11th-Order Learner',  '13th-Order Learner', 'Location', 'northeastoutside');

%% (d) Calculate the mean squared error (MSE) associated with each of your
 ...learned models on the training data.

% changing orders to be 1:2:13
orders = 1:2:13;

% Add linear learner to the array of learners
learners = [learnerLinear, learners];

errorsTr = zeros(1,7);
for i = 1:7
	errorsTr(i) = mse(learners(i), polyx(Xtr, orders(i)), Ytr);
end

figure;
plot(orders, errorsTr, '-rx', 'MarkerSize', 10);
title('MSE vs Order of Polynomial');
xlabel('Learner Polynomial Order');
ylabel('Error');
legend('Training Data', 'Location', 'northeastoutside');

%% (e) Calculate the MSE for each model on the test data (in mTestData.txt ).

errorsTe = zeros(1,7);
for i = 1:7
	errorsTe(i) = mse(learners(i), polyx(Xte, orders(i)), Yte);
end

hold on
plot(orders, errorsTe, '-gx', 'MarkerSize', 10);
hold off

title('MSE vs Order of Polynomial');
xlabel('Learner Polynomial Order');
ylabel('Error');
legend('Training Data', 'Testing Data', 'Location', 'northeastoutside');

%% (f) Calculate the MAE for each model on the test data. Compare the
...obtained MAE values with the MSE values obtained in above (e).

msError = zeros(1,7);
maError = zeros(1,7);

for i = 1:7
	msError(i) = mse(learners(i), polyx(Xte, orders(i)), Yte);
	maError(i) = mae(learners(i), polyx(Xte, orders(i)), Yte);
end

figure;
hold on
plot(orders, msError, '-rx', 'MarkerSize', 10);
plot(orders, maError, '-bx', 'MarkerSize', 10);
hold off

title('MSE vs MAE');
xlabel('Learner Polynomial Order');
ylabel('Error');
legend('MSE', 'MAE', 'Location', 'northeastoutside');





