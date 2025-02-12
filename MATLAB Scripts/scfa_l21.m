% The purpose of this algorithm is to solve: 
% min(W) f(W) = tr(WtAW) + 2tr(WtB) + lambda ||W||_21
% where A(nxn) is an PSD matrix, B(nxk) is arbitrary
% W(nxk) (we assume that k <= n) with orthonormal columns
% and lambda > 0.  

% Input
% A:  PSD matrix of size n x n
% B:  n x k matrix 
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.init = 0 or 1 (0 for no initial W and 1 for initial provided)
    % opts.W = initial W of size n x k(with orthonormal columns, and assuming opts.init = 1)
    % opts.lambda = lambda > 0 used in f(W).  


% info:
% info.f = 1 x ___ array of objective values f(W) over each iteration
% info.res = 2 x ____ array of normalized residual errors with the 
%            first entry measuring ||AW+B + lambda*DW-W*Lambda||_F/(||A||_F + ||B||_F + lambda*n)
%            where Lambda = WtAW + WtB + lambda WtDW.  
%            and the second measuring ||WtB - BtW||_F/(||B||_F)
% info.time = cputime (secs) from start to finish of algorithm
% info.W = most recently W computed 

function info = scfa_l21(A,B,opts)

% Start time
time = cputime; 

% Initilizing constants
[n,k] = size(B); 
lambda = opts.lambda; 
two_lambda = 2*lambda; 

% For interior SCF function
opts_in.lambda = lambda; 
opts_in.maxit = 500; % Arbitrary value (chose it to be less than outer maxit)

% Initialization diagonal vecD
vecD = zeros(n,1); 

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
nrm_kkt = nrmA + nrmB + opts.lambda*n;

% Constructing matrices for objective function and KKT errors
AW = A*W; WAW = W'*AW; 
for i = 1:n
    vecD(i) = 0.5/norm(W(i,:)); 
end
DW = W.*vecD; WDW = W'*DW; 
WtB = W'*B; 

% Gradient G and Lambda L.  
G = AW+B+lambda*DW; 
L = WAW+WtB+lambda*WDW; 

% Residual matrix R (note that R is orthogonal to W).
R = G - W*L; 

% Calculating current objective function value
f = trace(WAW)+2*trace(WtB)+two_lambda*trace(WDW); f_all = f; 

% Calculating current individual KKT errors
res_kkt = norm(R,'fro')/nrm_kkt; 
res_sym = norm(WtB-WtB','fro')/nrmB;

% Calculating current total KKT errors and saving info 
res_err = res_kkt + res_sym; 
res_all = [res_kkt res_sym]; 

iter = 0; Wp = W; 

while (res_err > opts.tol) && (opts.maxit > iter)
    
    % Finding a matrix Q such that Q is an orthonormal 
    % basis matrix of [W R Wp] and the first k columns 
    % of Q is W
    if iter >= 1
        R = [R Wp-W*(W'*Wp)];
    end
    R = orth(R); R = R-W*(W'*R); 
    R = orth(R); Q = [W R];  
    
    % Projecting onto the subspace spanned by the columns of Q = [W R]
    AR = A*R; AQ = [AW AR]; WAR = W'*AR;
    Atilde = [WAW WAR; WAR' R'*AR]; Btilde = [WtB; R'*B]; 
     
    % Make next approximation slightly better
    % You can play around with the percentage
    % Maybe 0.01*res_err or 0.25*res_err
    % I haven't found a single "best" percentage, 
    % but 0.1*res_err works well.
    opts_in.tol = 0.25*res_err;  
 
    % Solve internal subproblem
    opts_in.P = eye(size(Btilde)); 
    P = scf_for_scfa_l21(Atilde,Btilde,Q,opts_in); 
    
    % Redefining matrices for next iteration
    Wp = W; 
    W = Q*P; AW = AQ*P; 

    % Constructing matrices for objective function and KKT errors
    WAW = W'*AW; 
    for i = 1:n
        vecD(i) = 0.5/norm(W(i,:)); 
    end
    DW = W.*vecD; WDW = W'*DW; 
    WtB = W'*B; 
    
    % Gradient G and Lambda L.  
    G = AW+B+lambda*DW; 
    L = WAW+WtB+lambda*WDW; 

    % Calculating f
    f = trace(WAW)+2*trace(WtB)+2*lambda*trace(WDW); 
    f_all = [f_all;f];
    
    % Residual matrix R (note that R is orthogonal to W).
    R = G - W*L; 
    
    % Calculating current individual KKT errors
    res_kkt = norm(R,'fro')/nrm_kkt; 
    res_sym = norm(WtB-WtB','fro')/nrmB;
    
    % Calculating current total KKT errors and saving info 
    res_err = res_kkt + res_sym; 
    res_all = [res_all;res_kkt res_sym];  

    iter = iter + 1; 

end

info.time = cputime - time; 
info.f = f_all; 
info.res = res_all; 
info.W = W; 


end