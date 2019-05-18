%demo_hpelm.m
% A demo of HPELM for MNIST Classiffication
%========================================================================== 
% paper:Huang G, Bai Z, Kasun L, et al. Local Receptive Fields Based 
%   Extreme Learning Machine[J]. Computational Intelligence Magazine IEEE, 
%   2015, 10(2):18 - 29.
%
% myblog:http://blog.csdn.net/enjoyyl/article/details/45724367
%==========================================================================
%
% ---------<Liu Zhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<http://blog.csdn.net/enjoyyl>
% ---------<https://www.linkedin.com/in/%E5%BF%97-%E5%88%98-17b31b91>
% ---------<2015/11/24>
% 

clear all;

%% load MNIST data
disk = 'D:/';
disk = '/mnt/d/';
% disk = '/home/user/zhi/';

data = load([disk, '/DataSets/dgi/MNIST/mnist_uint8.mat']);

train_x = double(reshape(data.train_x',28,28,60000))/255.0;
train_y = data.train_y;

test_x = double(reshape(data.test_x',28,28,10000))/255.0;
test_y = data.test_y;
clear data


%% Setup ELM-LRF

hpelm.layers = {
	struct('type', 'i') %input layer
	struct('type', 'c', 'outputmaps', 24, 'kernelsize', 9) %convolution layer
%     struct('type', 'c', 'outputmaps', 4, 'kernelsize', 5) %convolution layer
	struct('type', 's', 'scale', 5) %sub sampling layer
	struct('type', 's', 'scale', 3) %sub sampling layer
    struct('type', 's', 'scale', 5) %sub sampling layer
    struct('type', 's', 'scale', 3) %sub sampling layer
    struct('type', 's', 'scale', 5) %sub sampling layer
    struct('type', 's', 'scale', 3) %sub sampling layer
    struct('type', 's', 'scale', 7) %sub sampling layer    
};


opts.isUseClassDistFuzzy = 0;
opts.isUseTrainErrorFuzzy = 0;
opts.isUseRandErrorFuzzy = 0;
opts.batchsize = 10000;
% opts.model = 'squexc';
% opts.model = 'squeeze';
opts.model = 'sequential';
% opts.model = 'parallel';
opts.randseed = [];
opts.randseed = 0;
opts.activation = [];
% opts.activation = 'relu';
% opts.activation = 'tanh';

for i=1:numel(hpelm.layers)
    disp(hpelm.layers{i})
end
disp(opts)

% setup
hpelm = hpelmsetup(hpelm, train_x, opts);

Cs = [ 0.1 0.001 0.01 0.1 0.2 0.3 0.4 0.5];
Nss = [20000, 1000, 10000, 20000, 30000];

for Ns = Nss
    opts.batchsize = min(Ns, opts.batchsize);

for C = Cs
	opts.C = C; 
    fprintf('\n=====With C  = %f,  Ns = %f=====\n', opts.C, Ns);

	%% training of ELM-LRF
    [hpelm, er, training_time] = hpelmtrain(hpelm, train_x(:,:,1:Ns,:), train_y(1:Ns,:), opts);

	% disp training error
	 fprintf('\nTraining error: %f\nTraining Time:%fs\n', er, training_time);

	%% Test ELM-LRF
	% disp testing error
	[er, bad, testing_time] = hpelmtest(hpelm, test_x, test_y, opts);

	fprintf('\nTesting error: %f\nTesting Time:%fs\n', er, testing_time);
    
end
end