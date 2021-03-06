function [ er, bad, testing_time ] = hpelmtest( net, x, y, opts )
%ELMLRFTEST Test ELM-LRF
%   

%==========================================================================
% Developed based on "cnn" of "DeepLearnToolbox" of rasmusbergpalm on GitHub
%   https://github.com/rasmusbergpalm/DeepLearnToolbox
%   
%==========================================================================
% ---------<LiuZhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<2015/11/24>
%==========================================================================
%
testing_time = cputime;

%forward
% model
hpelmf = str2func(['@hpelmf_' opts.model]);
net = hpelmf(net, x, opts);
clear x

predT = net.h * net.BETA; % (N, K(d-r+1)) * (K(d-r+1),nClasses)

[~, label0] = max(y, [], 2);
[~, label] = max(predT, [], 2);

bad = find(label0 ~= label);
er = numel(bad) / size(y, 1);  
testing_time = cputime - testing_time;
end

