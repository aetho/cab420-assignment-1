%% Setup
clear; clc; close all;

iris=load( 'data/iris.txt' );	% load the text file
X = iris(:,1:2); Y=iris(:,end);	% get first two features
[X, Y] = shuffleData(X,Y);		% reorder randomly
X = rescale(X);					% works much better for rescaled data
XA = X(Y<2,:); YA=Y(Y<2);		% get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0);		% get class 1 vs 2

%% (a) Show the two classes in a scatter plot and verify that one is
...linearly separable while the other is not.
	
% Plotting class 0 vs 1
figure('Name', 'Linearly Separable');
hold on;
title('Plotting classes 0 and 1');
plot(XA(YA==0,1), XA(YA==0,2), 'ro', 'DisplayName', 'y = 0');
plot(XA(YA==1,1), XA(YA==1,2), 'go', 'DisplayName', 'y = 1');
axis([-3 3, -3, 3]);
xlabel('x_1');
ylabel('x_2');
legend show;
hold off;


% Plotting class 1 vs 2
figure('Name', 'Linearly Inseparable');
title('Plotting classes 1 and 2');
hold on;
plot(XB(YB==1,1), XB(YB==1,2), 'go', 'DisplayName', 'y = 1');
plot(XB(YB==2,1), XB(YB==2,2), 'bo', 'DisplayName', 'y = 2');
axis([-3 3, -3, 3]);
xlabel('x_1');
ylabel('x_2');
legend show;
hold off;

% COMMENT
%	Figure 1 show linearly separable data while figure 2 shows linearly
%	inseparable data.
%

%% (b) Write ( fill in) the functio n @logisticClassify2/plot2DLinear.m
...so that it plots the two classes of data in different colors, along with
...the decision boundary (a line). Include the listing of your code in your
...report.

learner = logisticClassify2(); % create "blank" learner
wts = [0.5 1 -0.25];
learner = setWeights(learner, wts); % set the learner's parameters

learner = setClasses(learner, unique(YA)); % define class labels using YA
plot2DLinear(learner, XA, YA);
title('plot2DLinear demo (Data A)');

learner = setClasses(learner, unique(YB)); % define class labels using YB
plot2DLinear(learner, XB, YB);
title('plot2DLinear demo (Data B)');

%% (c) Complete the predict.m function to make predictions for your
...linear classifier.
	
learner = logisticClassify2();					% create "blank" learner
learner = setWeights(learner, [0.5 1 -0.25]);	% set the learner's parameters

% Calculating error in data set A
learner = setClasses(learner, unique(YA));		% define class labels using YA
YteA = predict(learner, XA);					% Use learner to predict XA

wrMaskA = (YteA ~= YA);	% Wrong Mask
errA = sum(wrMaskA)/length(wrMaskA);
disp(['Error rate of model on data set A: ', num2str(errA)]);

% Calculating error in data set B
learner = setClasses(learner, unique(YB));		% define class labels using YB
YteB = predict(learner, XB);					% Use learner to predict XB

wrMaskB = (YteB ~= YB);	% Wrong Mask
errB = sum(wrMaskB)/length(wrMaskB);
disp(['Error rate of model on data set B: ', num2str(errB)]);

%% (d) DONE

%% (e) DONE
% Implemented in train.m

%% (f) Setup
clear; clc; close all;
iris=load( 'data/iris.txt' );	% load the text file
X = iris(:,1:2); Y=iris(:,end);	% get first two features
[X, Y] = shuffleData(X,Y);		% reorder randomly
X = rescale(X);					% works much better for rescaled data
XA = X(Y<2,:); YA=Y(Y<2);		% get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0);		% get class 1 vs 2

%% (f1) Run logistic regression classifier on data set A
close all; clc;

learnerA = logisticClassify2(XA, YA,...		% Train learner on data set A
	'reg', 0,...				% no regularization (alpha = 0)
	'stepsize', 1,...			% Step size = 1
	'stoptol', 0.0001,...		% Stop tolerance = 0.0001
	'plot', false...			% Plot true
);

title('Convergence Surrogate Loss and Error rate (Data A)');
xlabel('Iteration');
legend('Surrogate Loss', 'Error Rate');

% Plot final converged classifier
figure;
plot2DLinear(learnerA, XA, YA);
title('Converged Logistic Classifier With Data A');



%% (f2) Run logistic regression classifier on data set B
close all; clc;

learnerB = logisticClassify2(XB, YB,...		% Train learner on data set B
	'reg', 0,...				% no regularization (alpha = 0)
	'stepsize', 1,...			% Step size = 1
	'stoptol', 0.0001,...		% Stop tolerance = 0.0001
	'plot', false...			% Plot true
);

title('Convergence Surrogate Loss and Error rate (Data B)');
xlabel('iteration');
legend('Surrogate Loss', 'Error Rate');

figure;
plot2DLinear(learnerB, XB, YB);
title('Converged Logistic Classifier With Data B');

%% (g) Setup
clear; clc; close all;
iris=load( 'data/iris.txt' );	% load the text file
X = iris(:,1:2); Y=iris(:,end);	% get first two features
[X, Y] = shuffleData(X,Y);		% reorder randomly
X = rescale(X);					% works much better for rescaled data
XA = X(Y<2,:); YA=Y(Y<2);		% get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0);		% get class 1 vs 2

%% (g1) Run logistic regression classifier on data set A (batch version)
close all; clc;

learnerA = logisticClassify2(XA, YA,...		% Train learner on data set A
	'reg', 0,...				% no regularization (alpha = 0)
	'stepsize', 1,...			% Step size = 1
	'stoptol', 0.0001,...		% Stop tolerance = 0.0001
	'plot', false...			% Plot true
);

title('Convergence Surrogate Loss and Error rate (Data A)');
xlabel('Iteration');
legend('Surrogate Loss', 'Error Rate');

% Plot final converged classifier
figure;
plot2DLinear(learnerA, XA, YA);
title('Converged Logistic Classifier With Data A');

%% (g2) Run logistic regression classifier on data set B (batch version)
close all; clc;

learnerB = logisticClassify2(XB, YB,...		% Train learner on data set A
	'reg', 0,...				% no regularization (alpha = 0)
	'stepsize', 1,...			% Step size = 1
	'stoptol', 0.0001,...		% Stop tolerance = 0.0001
	'plot', false...			% Plot true
);

title('Convergence Surrogate Loss and Error rate (Data A)');
xlabel('Iteration');
legend('Surrogate Loss', 'Error Rate');

% Plot final converged classifier
figure;
plot2DLinear(learnerB, XB, YB);
title('Converged Logistic Classifier With Data B');


%% (g) DONE
% Implemented in create_mini_batches and train_in_batches.m







