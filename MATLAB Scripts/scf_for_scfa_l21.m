% The purpose of this algorithm is to solve the interior subproblem for SCFA: 
% min(P) f(P) = tr(Pt*Atilde*P) + 2tr(Pt*B) + lambda ||Q*P||_21
% where Atilde = Qt*A*Q and Btilde = Qt*B.  
% Q is an n x q (k <= q <= 3k) orthonormal matrix whose 
% first k columns is the most recently computed W from SCFA 
% A is an n x n PSD matrix (therefore Atilde is PSD and q x q) lambda > 0.
% B is an n x k matrix (Btilde is then q x k).

% Input
% Atilde:  PSD matrix of size q x q
% Btilde:  q x k matrix 
% Q:       n x k orthonormal matrix used to generate Atilde and Btilde.
% opts:
    % opts.tol = stopping tolerance (ex: 1e-4)
    % opts.maxit = max number of iterations (ex: 1000)
    % opts.P = initial P of size q x k and the first k rows of P are the
    %          identity matrix and the rest is zeros.
    % opts.lambda = lambda > 0 used in f(P).  


% Output:  most recently computed P

% KKT Errors
%            ||AtildeP+Btilde + lambda*DtildeP-P*Lambda||_F/(||Atilde||_F + ||Btilde||_F + lambda*q)
%            where Lambda = PtAtildeP + Btilde + lambda PtDtildeP.  
%            and the second measuring ||PtBtilde - BtildetP||_F/(||Btilde||_F)


function P = scf_for_scfa_l21(Atilde,Btilde,Q,opts)

% Initializing constants
P = opts.P; lambda = opts.lambda; 
[q,k] = size(Btilde); n = size(Q,1); 

% initializing D (and equivalently Dtilde)
D = zeros(n,1); QP = Q*P; 
for i = 1:n 
    D(i) = 0.5/norm(QP(i,:));   
end

DQ = D.*Q; 
Dtilde = Q'*DQ; 

nrmB = norm(Btilde,'fro'); 
nrm_kkt = norm(Atilde,'fro')+nrmB+lambda*q;

% Calculating gradient G and lambda L
AtildeP = Atilde*P; PAtildeP = P'*AtildeP; PtBtilde = P'*Btilde; 
DtildeP = Dtilde*P; PDtildeP = P'*DtildeP; 

G = AtildeP + Btilde + lambda*DtildeP; 
L = PAtildeP + PtBtilde + lambda*PDtildeP; 

% Calculating residual matrix R
R = G-P*L; 

% Calculating intitial KKT errors and objective values
res_kkt = norm(R,'fro')/nrm_kkt; 
res_sym = norm(PtBtilde  - PtBtilde','fro')/nrmB; 
res_err = res_kkt + res_sym; 

iter = 0; 

while (opts.tol < res_err) && (opts.maxit > iter)
    
    % Create J(P) to solve NEPv:  J(P)P = P*Psy
    % where Psy = P'* J(P) * P and P is an orthonormal eigenbasis matrix
    % of J(P) associated with its k smallest eigenvalues.
    PBtildeT = P*Btilde'; 
    DPPt = Dtilde*(P*P'); 
    J = Atilde + (PBtildeT + PBtildeT') + lambda*(DPPt + DPPt'); 
    % J = Atilde + (PBtildeT + PBtildeT') + 2*lambda*Dtilde; 
    J = (J+J')./2; 
    [U,Evals] = eig(J); Evals = diag(Evals); [~,idx] = sort(Evals,'ascend');
    P = U(:,idx(1:k)); 

    % Refinement step (Update P <- -PUVt)
    PtBtilde = P'*Btilde; 
    [U,~,V] = svd(PtBtilde); 
    UVt = -U*V'; 
    P = P*UVt; 
    
    % Begin recreating matrices (especially QP)
    PtBtilde = UVt'*PtBtilde; 
    AtildeP = Atilde*P; PAtildeP = P'*AtildeP; 
    QP = Q*P; 

    % Recreating D matrices
    for i = 1:n
        D(i) = 0.5/norm(QP(i,:)); 
    end
    DQ = D.*Q; 
    Dtilde = Q'*DQ;
    DtildeP = Dtilde*P; 
    PDtildeP = P'*DtildeP; 

    % Matrix = AP + B - P*Lambda
    % Lambda = P'*AP + P'B; 

    % || AP+B-P*( P'*AP + P'B)||_F / (||A|| + ||B||)
    
    % Recalculating gradient G and lambda L
    G = AtildeP + Btilde + lambda*DtildeP; 
    L = PAtildeP + PtBtilde + lambda*PDtildeP; 
    
    % Calculating residual matrix R
    R = G-P*L; 
    
    % Calculating current KKT errors
    res_kkt = norm(R,'fro')/nrm_kkt; 
    res_sym = norm(PtBtilde  - PtBtilde','fro')/nrmB; 
    res_err = res_kkt + res_sym; 

    iter = iter + 1; 
end

end