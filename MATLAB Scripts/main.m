% clear;  
% clc;    
% close all;  

% 数据加载
% load('digit_data.mat'); % 得到X (2000 x 643)和y (1 x 2000)

% 数据预处理：合并所有视角并中心化
X_concat = normalize(X)';
d = size(X_concat, 1);

% 参数设置
target_m = 8;         % 目标维度
max_iter = 50;  % 最大外层迭代次数
tol = 1e-4;     % 外层收敛阈值

% 初始化P: d × target_m
P_inits = randn(d, target_m);
P_inits = orth(P_inits);

% opts参数设置（用于OptimizeP_scfa）
opts = struct();
opts.tol = 1e-5;      % 收敛容差
opts.maxit = 500;    % 最大迭代次数
opts.init = 1;        % 使用随机初始化
opts.k = target_m;    % 设置SCFA的目标维度
% opts.gamma = 0.01;    % L2,1范数系数
% opts.mu = 0.1;        % S的正则化系数
opts.W = P_inits; 

% params参数设置（用于OptimizeS）
params = struct();
params.k = 10;           % k近邻数量
params.sigma = 3.0;      % 高斯核参数
params.parallel = true;  % 是否使用并行计算
params.verbose = true;   % 是否显示统计信息

% Overall参数设置
lambda = 1000;
opts.lambda = lambda; 


% 计时开始
tic;

% 调用主算法
[P_final, S_final, n_iter] = Overall(X_concat, P_inits, opts, params, lambda, max_iter, tol);

% 计时结束
computation_time = toc;

% 输出性能统计
disp('Algorithm Performance Statistics:');
disp(['Computation Time: ', num2str(computation_time), ' seconds']);
fprintf('Algorithm converged at iteration %d\n', n_iter);

% 输出投影矩阵信息
disp('Projection Matrix Statistics:');
disp(['Dimensions: ', num2str(size(P_final))]);
disp(['Condition Number: ', num2str(cond(P_final))]);
disp(['Rank: ', num2str(rank(P_final))]);

% 输出相似度矩阵信息
disp('Similarity Matrix Statistics:');
disp(['Sparsity: ', num2str(nnz(S_final)/numel(S_final))]);
disp(['Maximum Value: ', num2str(max(S_final(:)))]);
disp(['Minimum Value: ', num2str(min(S_final(:)))]);
disp(['Mean Value: ', num2str(mean(S_final(:)))]);

% 可视化结果
figure('Name', 'Algorithm Analysis');

% 相似度矩阵热图
subplot(2,2,1);
imagesc(S_final);
colorbar;
title('Similarity Matrix Heatmap');
xlabel('Sample Index');
ylabel('Sample Index');

% 投影矩阵热图 
subplot(2,2,2);
imagesc(P_final);
colorbar;
title('Projection Matrix Heatmap');
xlabel('Reduced Dimension');
ylabel('Original Feature Dimension');

% 使用t-SNE可视化投影后的数据
projected_data = P_final' * X_concat;
subplot(2,2,3);
Y = tsne(projected_data', 'NumDimensions', 2);
scatter(Y(:,1), Y(:,2), 20, 'filled');
title('t-SNE Visualization of Projected Data');
xlabel('First Dimension');
ylabel('Second Dimension');

% 保存结果
results = struct();
results.P = P_final;
results.S = S_final;
results.projected_data = projected_data;
results.tsne_coords = Y;  % 保存t-SNE坐标
save('results.mat', '-struct', 'results');