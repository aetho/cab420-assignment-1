function obj = logisticClassify2(Xtr,Ytr, varargin)
% logisticClassify(X,Y,...) : construct a logistic classifier (linear classifier with saturated output)
% can take no arguments, or see logisticClassify/train for training options

  obj.wts=[];         % linear weights on features (1st weight is constant term)
  obj.classes=[];     % list of class values used in input
  obj=class(obj,'logisticClassify2');
  if (nargin > 0) 
	% -- OUR CODE -- %
	% Train using single data point j 
    % obj=train(obj,Xtr,Ytr, varargin{:});
	
	% Train using mini batches sized 11
	obj = train_in_batches(obj, Xtr, Ytr, 11, varargin{:});
	% -- END OF OUR CODE -- %
  end;
end

