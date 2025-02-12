% The purpose of this algorithm is to solve: min(W) f(W) = tr(WtAW) + 2tr(WtB)
% where A(nxn) is an PSD matrix, B(nxk) is arbitrary
% and W(nxk) (we assume that k <= n)

% This algorithm turns the minimization problem into a maximization problem
% of the form:  max(W) F(W) = tr(Wt Ahat W) + 2tr(Wt Bhat)
% where Ahat = alpha I - A and Bhat = -B
% alpha > 0 is any value such that Ahat is PSD.

% Input
% A:  PSD matrix of size n x n
% B:  n x k matrix 
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k(with orthonormal columns, and assuming opts.init = 1)
    % opts.alpha = positive constant such that Ahat = alpha I - A is PSD 
        % (1 norm of A works well)


% info:
% info.f = 1 x ___ array of objective values f(W) over each iteration
% info.res = 2 x ____ array of normalized residual errors with the 
%            first entry measuring ||AhatW+Bhat-What*Lambda||_F/(||Ahat||_F + ||Bhat||_F)
%            and the second measuring ||WtBhat - BhattW||_F/(||Bhat||_F)
% info.time = cputime (secs) from start to finish of algorithm
% info.W = most recently W computed 


function info = gpi(A,B,opts)

% Start time
time = cputime; 

% Size of B
[n,k] = size(B); 

% Create Ahat and Bhat
Ahat = opts.alpha*eye(n) - A; Bhat = -B; 

% Initializing W 
if opts.init == 1
    W = opts.W; 
elseif opts.init == 0
    W = orth(rand(n,k)); 
else
    print('Invalid input for opts.init')
    return 
end

% One time calculations
nrmAhat = norm(Ahat,'fro'); nrmBhat = norm(Bhat,'fro'); 
nrm_kkt = nrmAhat + nrmBhat; 

% Constructing matrices for objective function and KKT errors
AhatW = Ahat*W; WAhatW = W'*AhatW; WtBhat = W'*Bhat; 
L = WAhatW+WtBhat; 

% Calculating current objective function value
alpha_k = opts.alpha*k; 
f = alpha_k-(trace(WAhatW)+2*trace(WtBhat)); f_all = f; 

% Calculating current individual KKT errors
res_kkt = norm(AhatW+Bhat-W*L,'fro')/nrm_kkt; 
res_sym = norm(WtBhat-WtBhat','fro')/nrmBhat;

% Calculating current total KKT errors and saving info 
res_err = res_kkt + res_sym; 
res_all = [res_kkt res_sym]; 

iter = 0; 

while (res_err > opts.tol) && (iter < opts.maxit)

    M = 2*(AhatW+Bhat); 
    [U,~,V] = svd(M,'econ'); 
    W = U*V'; 
    
    % Constructing matrices for objective function and KKT errors
    AhatW = Ahat*W; WAhatW = W'*AhatW; WtBhat = W'*Bhat; 
    L = WAhatW+WtBhat; 
    
    % Calculating current objective function value
    f = alpha_k-(trace(WAhatW)+2*trace(WtBhat)); 
    f_all = [f_all;f]; 
    
    % Calculating current individual KKT errors
    res_kkt = norm(AhatW+Bhat-W*L,'fro')/nrm_kkt; 
    res_sym = norm(WtBhat-WtBhat','fro')/nrmBhat;
    
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