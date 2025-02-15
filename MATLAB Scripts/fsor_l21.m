% This function solves the FSOR-l21 problem via SCFA-l21
% Find min(W) tr(W^T XX^T W) + 2tr(W^T(-XY^T)) + ||W||_21
% where W lives on the Stiefel Manifold (n x k)

% A = X*X' and B = -X*Y'

% Input variables
% X - mean centered data matrix of size n x m (n features, m samples)
% Y - mean cenetered label matrix of size k x m (k classes, m samples)
% opts - options for SCFA-l21
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k (with orthonormal columns, and assuming opts.init = 1)
    % opts.lambda = lambda > 0 used in f(W).  


function info = fsor_l21(X,Y,opts)

% start time 
time = cputime; 

% construct matrices A and B
A = X*X'; B = -X*Y'; 
scfa_info = scfa_l21(A,B,opts); 

% Pre-allocate space for weights
n = size(A,1); weights = zeros(n,1);

% Creating vector of weights measuring the importance of 
% each feature of X.
W = scfa_info.W; 
for i = 1:n
    weights(i) = norm(W(i,:)); 
end

% Now the total sum of weights add up to 1
% If index j has the largest value, then feature j is considered the most valuable/important 
% and has strong predictive power.  Similar logic for the feature with the smallest weight.
weights = weights./sum(weights);

% output data
info.time = cputime - time; 
info.f = scfa_info.f; 
info.W = scfa_info.W; 
info.weights = weights; 
info.res = scfa_info.res; 






end
