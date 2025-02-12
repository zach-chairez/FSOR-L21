% The purpose of this algorithm is to solve: min(W) f(W) = tr(WtAW) + 2tr(WtB)
% where A(nxn) is an PSD matrix, B(nxk) is arbitrary
% and W(nxk) (we assume that k <= n)

% Input
% A:  PSD matrix of size n x n
% B:  n x k matrix 
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k(with orthonormal columns, and assuming opts.init = 1)


% info:
% info.f = 1 x ___ array of objective values f(W) over each iteration
% info.res = 2 x ____ array of normalized residual errors with the 
%            first entry measuring ||AW+B-W*Lambda||_F/(||A||_F + ||B||_F)
%            and the second measuring ||WtB - BtW||_F/(||B||_F)
% info.time = cputime (secs) from start to finish of algorithm
% info.W = most recently W computed 


function info = scf_refined(A,B,opts)

% Start time
time = cputime; 

% Size of B
[n,k] = size(B); 

% Initializing W 
if opts.init == 1
    W = opts.W; 
elseif opts.init == 0
    W = orth(rand(n,k)); 
else
    error('Invalid input for opts.init - need to be 0 or 1')
end

% One time calculations
nrmA = norm(A,'fro'); nrmB = norm(B,'fro'); 
nrm_kkt = nrmA + nrmB; 

% Initial refinement step
WtB = W'*B; 
[U,~,V] = svd(WtB); 
P = -U*V'; WtB = P'*WtB; 
W = W*P; 

% Constructing matrices for objective function and KKT errors
AW = A*W; WAW = W'*AW; 
L = WAW+WtB; 

% Calculating current objective function value
f = trace(WAW)+2*trace(WtB); f_all = f; 

% Calculating current individual KKT errors
res_kkt = norm(AW+B-W*L,'fro')/nrm_kkt; 
res_sym = norm(WtB-WtB','fro')/nrmB;

% Calculating current total KKT errors and saving info 
res_err = res_kkt + res_sym; 
res_all = [res_kkt res_sym]; 

iter = 0; 

while (res_err > opts.tol) && (iter < opts.maxit)
    
    % Calculate next approximate W
    WBt = W*B'; 
    E = A + (WBt+WBt'); E = (E+E')./2; 
    [W,~] = eigs(E,k,'smallestreal'); 

    % Refinement
    WtB = W'*B; 
    [U,~,V] = svd(WtB); 
    P = -U*V'; WtB = P'*WtB; 
    W = W*P; 
    
    % Constructing matrices for objective function and KKT errors
    AW = A*W; WAW = W'*AW;  
    L = WAW+WtB; 
    
    % Calculating current objective function value
    f = trace(WAW)+2*trace(WtB); 
    f_all = [f_all;f]; 
    
    % Calculating current individual KKT errors
    res_kkt = norm(AW+B-W*L,'fro')/nrm_kkt; 
    res_sym = norm(WtB-WtB','fro')/nrmB;
    
    % Calculating current total KKT errors and saving info 
    res_err = res_kkt + res_sym; 
    res_all = [res_all;res_kkt res_sym]; 
    
    % Increase iteration number
    iter = iter + 1; 


end

info.time = cputime - time; 
info.f = f_all;
info.res = res_all; 
info.W = W; 




end
