function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

	% -- OUR CODE -- %
	% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
	wts = obj.wts;
	f = wts(1) + wts(2)*Xte(:,1) + wts(3)*Xte(:,2);
	fSign = sign(f);
	
	% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );
	yh = zeros(size(Xte,1), 1);

	negMask = fSign < 0;
	posMask = fSign > 0;
	
	yh(negMask) = obj.classes(1);
	yh(posMask) = obj.classes(2);
	
	Yte = yh;
	% -- END OF OUR CODE -- %
end