function [Band,W2,miTrain,y_train] = fbcspRe(S1,m) % Following paper "Common Spatial Pattern Method for Channel Selection in
% Motor Imagery Based Brain-Computer Interface" by Wang et al - [1]
%% Average covariance of Left and Right
Cre = zeros(size(S1(1,1).Re,2),size(S1(1,1).Re,2),size(S1,2));
Clr = zeros(size(S1(1,1).LR,2),size(S1(1,1).LR,2),size(S1,2));
featRe = zeros(size(S1,1),9*m*2);
featLR = zeros(size(S1,1),9*m*2);
bp = 4:4:40; 
for i = 1:size(S1,2)
    for j = 1:size(S1,1)
        Cre(:,:,i) = Cre(:,:,i) + covar(S1(j,i).Re');
        Clr(:,:,i) = Clr(:,:,i) + covar(S1(j,i).LR');
    end
end

Cre = Cre ./ size(S1,2);
Clr = Clr ./ size(S1,2);

for i = 1:size(S1,2)
    R(:,:,i) = Cre(:,:,i) + Clr(:,:,i); % (2)
    %% Eigen decomp of R (2)
    [V(:,:,i),D(:,:,i)] = eig(R(:,:,i)); % V = U and D = Sigma as denoted in (2)
    %% Whitening Matrix: P = inverse square root(D) * U' (3)
    P(:,:,i) = sqrtm(inv(D(:,:,i))) * V(:,:,i)';
    %% Sl = P*Cl*P'  Sr = P*Cr*P' (4)
    sRe(:,:,i) = P(:,:,i)*Cre(:,:,i)*P(:,:,i)';
    sLR(:,:,i) = P(:,:,i)*Clr(:,:,i)*P(:,:,i)';
    
    %% If we eig decomp sL and sR, the sum of eigenvalues should be an
    % identity matrix of NxN. (5)
    [sreV2(:,:,i),sreD2(:,:,i)] = decompose_S(sRe(:,:,i),'ascending');
    [slrV2(:,:,i),slrD2(:,:,i)] = decompose_S(sLR(:,:,i),'descending');
end
check = slrD2+sreD2; % All are 1, which means I.D Matrix confirmed. (5)
% if check ~= 1
%     error('I.D Matrix not confirmed');
% end

% Projection Matrix: W = U'*P where U' is slV2 and P is whitening matrix (6)
for i = 1:size(S1,2)
    W(:,:,i) = sreV2(:,:,i)' * P(:,:,i); %(6)
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
        S1(j,i).Zre = W2(:,:,i) * S1(j,i).Re';
        S1(j,i).Zlr = W2(:,:,i) * S1(j,i).LR';
    end
    
    % Feature vector array. Trial X Bandwidth: 72 x 36 where 1:4 is 4-8Hz,
    % 5-8 is 8-12Hz and so on.
    for j = 1:size(S1,1)
        var1 = var(S1(j,i).Zre, 1, 2);
        varsum = sum(var1);
        featRe(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum); % -3 -4
        clear var1 varsum
        
        var1 = var(S1(j,i).Zlr, 1, 2);
        varsum = sum(var1);
        featLR(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum
    end
    
end

% Adding True label and mergin feature vector
% TRUE LABEL
true_rest = ones(size(featRe, 1), 1);
true_lr = zeros(size(featLR, 1), 1);

% FEATURE + TRUE LABEL
rest = [featRe, true_rest];
lr = [featLR, true_lr];

% MERGE LEFT AND RIGHT
train_feat = [rest; lr];

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
