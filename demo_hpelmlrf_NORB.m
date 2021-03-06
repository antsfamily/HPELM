%demo_hpelm.m
% A demo of HPELM for NORB Classiffication
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

%% load NORB data

disk = 'D:/';
disk = '/mnt/d/';
% disk = '/home/user/zhi/';

% for training
load([disk, '/DataSets/oi/nsi/NORB/norb_traindata.mat']); %X is H*W*C-N, Y is N-1

train_x = reshape(X, 32,32,2,size(X,2));% X is H*W*C-N --> H-W-C-N
train_x = permute(train_x, [1 2 4 3]); % H-W-N-C
train_y = full(sparse(1:size(Y,1),Y,1)); % Y is N-1  -->  N*nClasses

% for testing
load([disk, '/DataSets/oi/nsi/NORB/norb_testdata.mat']);
% a = min(X(:));
% b = max(X(:));
% c = 0;
% d = 1;
% X = (X-a)*(d-c)/(b-a) + c;
test_x = reshape(X, 32, 32, 2, size(X,2));
test_x = permute(test_x, [1 2 4 3]);
test_y = full(sparse(1:size(Y, 1), Y, 1));
clear X Y;

%% Setup ELM-LRF

hpelm.layers = {
	struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 28, 'kernelsize', 4) %convolution layer
	struct('type', 's', 'scale', 7) %sub sampling layer, 7 is 3 in the paper
%     struct('type', 'c', 'outputmaps', 18, 'kernelsize', 4) %convolution layer
	struct('type', 's', 'scale', 5) %sub sampling layer
%     struct('type', 'c', 'outputmaps', 64, 'kernelsize', 4) %convolution layer
	struct('type', 's', 'scale', 3) %sub sampling layer
    struct('type', 's', 'scale', 5) %sub sampling layer
%     struct('type', 's', 'scale', 3) %sub sampling layer
%     struct('type', 's', 'scale', 5) %sub sampling layer
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
opts.randseed = 2019;
opts.activation = [];
% opts.activation = 'relu';
% opts.activation = 'tanh';

for i=1:numel(hpelm.layers)
    disp(hpelm.layers{i})
end
disp(opts)

% setup
hpelm = hpelmsetup(hpelm, train_x, opts);

Cs = [ 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5];
Nss = [20000, 1000, 10000, 24300];

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
