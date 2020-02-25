%% 4. Nearest Neighbor Classifiers
%% Setup
clear; clc; close all;

iris = load( 'data\iris.txt' );			% load data
pi = randperm( size (iris ,1));			% Random permuation of row indices
Y = iris(pi ,5); X = iris(pi ,1:2);		% Retrieve Y, X based on pi

uniqueY = unique(Y);
plotCfg = {'bo', 'go', 'ro'};	% Plot configs for classes

%% (a) Plot the data by their feature values, using the class value to select the color

figure;
hold on;
for i = 1:length(uniqueY)
	mask = (Y == uniqueY(i));
	plot(X(mask, 1), X(mask, 2), plotCfg{uniqueY(i)+1},...
		'DisplayName', ['y = ', num2str(uniqueY(i))]);
end
legend('Location', 'northeastoutside');
legend show;
title('Plotting training data');
xlabel('x_1');
ylabel('x_2');
hold off;

%% (b) Use the provided knnClassify class to learn a 1-nearest-neighbor predictor.

k1Learner = knnClassify(1, X, Y);
class2DPlot(k1Learner,X,Y);
title('k = 1');
xlabel('x_1');
ylabel('x_2');

%% (c) Do the same thing for several values of k (say, [1, 3, 10, 30])
...and comment on their appearance.
	
kVals = [1 3 10 30];
kLearners(size(kVals)) = knnClassify;

for i = 1:length(kVals)
	kLearners(i) = knnClassify(kVals(i), X, Y);
	class2DPlot(kLearners(i), X, Y);
	title(['k = ' num2str(kVals(i))]);
	xlabel('x_1');
	ylabel('x_2');
end

% COMMENT
%	The decision boundary becomes simpler the for higher values of k
%	where as lower values of k result in more complex decision boundaries.
%

%% Setup
clear; clc; close all;

iris = load( 'data\iris.txt' );			% load data
pi = randperm( size (iris ,1));			% Random permuation of row indices
Y = iris(pi ,5); X = iris(pi ,1:2);		% Retrieve Y, X based on pi

uniqueY = unique(Y);
plotCfg = {'bo', 'go', 'ro'};	% Plot configs for classes

%% (d) split the data into an 80/20 training/validation split.
...For k = [1, 2, 5, 10, 50, 100, 200], learn a model on the 80% and
...calculate its performance (# of data classified incorrectly) on the
...validation data.

% Spliting data into 80/20 (train/test)
idx = round(size(iris, 1) * 0.2);

Xvalid = X(1:idx, :);
Yvalid = Y(1:idx, :);

Xtrain = X((idx+1):end, :);
Ytrain = Y((idx+1):end, :);

kVals = [1 2 5 10 50 100 200];
perf = zeros(size(kVals));		% Performance

for i = 1:length(kVals)
	kLearner =  knnClassify(kVals(i), Xtrain, Ytrain);
	Yhat = predict(kLearner, Xvalid);	% Predictions
	mask = (Yhat ~= Yvalid);			% Incorrect predictions mask
	perf(i) = sum(mask)/length(mask);	% Accuracy of learner
end

figure;
plot(kVals, perf, 'gx-');
title('Performance vs. k');
ylabel('Error rate');
xlabel('k');

% COMMENTS (This needs to change because randperm changes training data)
%	From the figure, it can be seen that k = 5:50 generalizes the best given
%	the training data. At the very left endpoint (k=1), the model seems to
%	overfit the data resulting a poor performance. For the very right
%	endpoint (k=200), the model seems to underfit the data and simply
%	predicts base on the majority class of the training data which also
%	results in a poorly performing model.






