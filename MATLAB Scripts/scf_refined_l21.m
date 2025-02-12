function info = scf_refined_l21(A,B,opts)

[n,ell] = size(B); 

if opts.init == 0
    W = orth(rand(n,ell)); 
elseif opts.init == 1
    W = opts.W; 
else
    error('opts.init can only takes the value 0 or 1'); 
end

D = zeros(n,1); 
for i = 1:n
    D(i) = 0.5/norm(W(i,:)); 
end

AW = A*W; WAW = W'*AW; 
WtB = W'*B; DW = D.*W; WDW = W'*DW; 

f = trace(WAW)+2*trace(WtB + opts.lambda*WDW); 
f_all = f; 

err_cond = true; iter = 0; 

while err_cond && (opts.maxit > iter)
    
    BWt = B*W'; 
    % DWWt = DW*W'; 
    % E = A + (BWt+BWt')+opts.lambda*(DWWt+DWWt');
    E = A + (BWt + BWt') + 2*opts.lambda*diag(D); 
    E = (E+E')./2; 
    [U,Evals] = eig(E); Evals = diag(Evals); [~,idx] = sort(Evals,'ascend');
    W = U(:,idx(1:ell)); 

    WtB = W'*B; 

    [U,~,V] = svd(WtB); 
    P = -U*V'; 
    W = W*P; 

    for i = 1:n
        D(i) = 0.5/norm(W(i,:)); 
    end


    AW = A*W; WAW = W'*AW; 
    WtB = W'*B; DW = D.*W; WDW = W'*DW; 
    
    f = trace(WAW)+2*trace(WtB + opts.lambda*WDW); 
    f_all = [f_all;f]; 
    
    rel_err = abs((f_all(end-1)-f)/f); 

    err_cond = rel_err > opts.tol; 

    iter = iter + 1; 
end

info.f = f_all; 
info.rel_err = rel_err; 





end