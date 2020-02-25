function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
	[n,d] = size(X);
	if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

	%%% TODO: Fill in the rest of this function...
	
	classes = unique(Y);		% Get classes from Y
	iY1 = (Y == classes(1));	% Get mask of Y == first class
	iY2 = (Y == classes(2));	% Get mask of Y == second class
	
	% Plotting data
	hold on;
	plot(X(iY1, 1), X(iY1, 2), 'ro');
	plot(X(iY2, 1), X(iY2, 2), 'go');
	
	% Plotting decision boundary
	wts = obj.wts;
	f = @(x1,x2) wts(1) + wts(2)*x1 + wts(3)*x2;
	fimplicit(f, axis, 'b');
	
	% Plot config
	xlabel('x_1');
	ylabel('x_2');
	legend('Class 0', 'Class 1', 'Decision Boundary',...
		'Location', 'northeastoutside'...
	);
	hold off;
end