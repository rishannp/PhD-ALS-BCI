function [Band,W2,miTrain,y_train] = fbcspR(S1,m) % Following paper "Common Spatial Pattern Method for Channel Selection in
% Motor Imagery Based Brain-Computer Interface" by Wang et al - [1]
%% Average covariance of Left and Right
Cr = zeros(size(S1(1,1).R,2),size(S1(1,1).R,2),size(S1,2));
Clre = zeros(size(S1(1,1).LRE,2),size(S1(1,1).LRE,2),size(S1,2));
featR = zeros(size(S1,1),9*m*2);
featLRE = zeros(size(S1,1),9*m*2);
bp = 4:4:40; 
for i = 1:size(S1,2)
    for j = 1:size(S1,1)
        Cr(:,:,i) = Cr(:,:,i) + covar(S1(j,i).R');
        Clre(:,:,i) = Clre(:,:,i) + covar(S1(j,i).LRE');
    end
end

Cr = Cr ./ size(S1,2);
Clre = Clre ./ size(S1,2);

for i = 1:size(S1,2)
    R(:,:,i) = Cr(:,:,i) + Clre(:,:,i); % (2)
    %% Eigen decomp of R (2)
    [V(:,:,i),D(:,:,i)] = eig(R(:,:,i)); % V = U and D = Sigma as denoted in (2)
    %% Whitening Matrix: P = inverse square root(D) * U' (3)
    P(:,:,i) = sqrtm(inv(D(:,:,i))) * V(:,:,i)';
    %% Sl = P*Cl*P'  Sr = P*Cr*P' (4)
    sR(:,:,i) = P(:,:,i)*Cr(:,:,i)*P(:,:,i)';
    sLRE(:,:,i) = P(:,:,i)*Clre(:,:,i)*P(:,:,i)';
    
    %% If we eig decomp sL and sR, the sum of eigenvalues should be an
    % identity matrix of NxN. (5)
    [srV2(:,:,i),srD2(:,:,i)] = decompose_S(sR(:,:,i),'ascending');
    [slreV2(:,:,i),slreD2(:,:,i)] = decompose_S(sLRE(:,:,i),'descending');
end
check = slreD2+srD2; % All are 1, which means I.D Matrix confirmed. (5)
% if check ~= 1
%     error('I.D Matrix not confirmed');
% end

% Projection Matrix: W = U'*P where U' is slV2 and P is whitening matrix (6)
for i = 1:size(S1,2)
    W(:,:,i) = srV2(:,:,i)' * P(:,:,i); %(6)
    % Computing Z: Z = WX (7)
    %    W = W();
    %    Z(:,:,i) =
    %    Z = [];
    %
    %m = 2; % First two columns and last 2 columns, like PCA choose most variant columns
    x = W(:,:,i);
    x([m+1:end-m],:) = [];
    W2(:,:,i) = x;
    for j = 1:size(S1,1)
        S1(j,i).Zr = W2(:,:,i) * S1(j,i).R';
        S1(j,i).Zlre = W2(:,:,i) * S1(j,i).LRE';
    end
    
    % Feature vector array. Trial X Bandwidth: 72 x 36 where 1:4 is 4-8Hz,
    % 5-8 is 8-12Hz and so on.
    for j = 1:size(S1,1)
        var1 = var(S1(j,i).Zr, 1, 2);
        varsum = sum(var1);
        featR(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum
        
        var1 = var(S1(j,i).Zlre, 1, 2);
        varsum = sum(var1);
        featLRE(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum
    end
    
end

% Adding True label and mergin feature vector
% TRUE LABEL
true_right = ones(size(featR, 1), 1);
true_rre = zeros(size(featLRE, 1), 1);

% FEATURE + TRUE LABEL
right = [featR, true_right];
rre = [featLRE, true_rre];

% MERGE LEFT AND RIGHT
train_feat = [right; rre];

% SHUFFLE TRAINING DATA
train_feat = train_feat(randperm(size(train_feat, 1)), :);

clearvars -except W2 train_feat S1 bp
%% Mutual Information based selection of Bands
x_train = train_feat(:,[1:end-1]);
y_train = train_feat(:,end);

for i = 1:size(x_train,2)
    % Compute histogram counts for bin and eeg
    N = size(x_train,2); % Number of bins
    cY = histcounts(y_train, N);
    cX = histcounts(x_train(:,i), N);
    
    % Compute joint histogram counts between phi_bin and spikeTrain
    cYX = histcounts2(y_train, x_train(:,i), N);
    
    % Convert counts to probabilities
    Py = cY / sum(cY);
    Px = cX / sum(cX);
    Pyx = cYX / sum(cYX);
    
    % Compute entropies
    Hy = -sum(Py(Py > 0) .* log2(Py(Py > 0)));
    Hx = -sum(Px(Px > 0) .* log2(Px(Px > 0)));
    Hyx = -sum(Pyx(Pyx > 0) .* log2(Pyx(Pyx > 0)));
    
    
    % Compute mutual information
    I(:,i) = Hy + Hx - Hyx;
end

% Find max 5 bands
[maxI,Band] = maxk(I,10);
miTrain = train_feat(:,[Band]);

end

