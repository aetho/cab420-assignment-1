function mini_batches = create_mini_batches(obj, X,y, batch_size )

n = length(y);

data_values = [X,y];
rIdx = randperm(n);

data_values = data_values(rIdx, :);
n_mini_batches = n/batch_size;
mini_batches = zeros(batch_size,3,n_mini_batches);

for i = 1:n_mini_batches
	bStart = (i-1)*batch_size + 1;
	bEnd = bStart + batch_size - 1;
	
	mini_batches(:, :, i) = data_values(bStart:bEnd, :);
end

end