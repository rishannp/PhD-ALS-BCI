clear all 
close all 
clc

load trainingData.mat

t0 = tic();


t1 = tic();
%% BPF and electrode reduction
[S1,S2,S3,S4,S5,S6,S7,S8] = preproc(P1,P2,P3,P4,P5,P6,P7,P8);
clear P1 P2 P3 P4 P5 P6 P7 P8
bpftime = toc(t1);
%% train
s = size(S1,1)*0.75;
s = round(s);

t2 = tic();

[x_train1,y_train1] = fbcsp(S1([1:s],:));
[x_train2,y_train2] = fbcsp(S2([1:s],:));
[x_train3,y_train3] = fbcsp(S3([1:s],:));
[x_train4,y_train4] = fbcsp(S4([1:s],:));
[x_train5,y_train5] = fbcsp(S5([1:s],:));
[x_train6,y_train6] = fbcsp(S6([1:s],:));
[x_train7,y_train7] = fbcsp(S7([1:s],:));
[x_train8,y_train8] = fbcsp(S8([1:s],:));

save('xyTrain');

%% test
s = size(S1,1)*0.75;
s = round(s);


[x_test1,y_test1] = fbcsp(S1([s:end],:));
[x_test2,y_test2] = fbcsp(S2([s:end],:));
[x_test3,y_test3] = fbcsp(S3([s:end],:));
[x_test4,y_test4] = fbcsp(S4([s:end],:));
[x_test5,y_test5] = fbcsp(S5([s:end],:));
[x_test6,y_test6] = fbcsp(S6([s:end],:));
[x_test7,y_test7] = fbcsp(S7([s:end],:));
[x_test8,y_test8] = fbcsp(S8([s:end],:));

clear S1 S2 S3 S4 S5 S6 S7 S8
save('xyTest');

fbcsptime = toc(t2);

totaltime = toc(t0);


%% Functions

function [miTrain,y_train] = fbcsp(S1) % Following paper "Common Spatial Pattern Method for Channel Selection in
% Motor Imagery Based Brain-Computer Interface" by Wang et al - [1]
%% Average covariance of Left and Right
Cl = zeros(22,22,size(S1(1,1).L,3));
Cr = zeros(22,22,size(S1(1,1).L,3));
featL = zeros(size(S1,1),4);
featR = zeros(size(S1,1),4);
bp = [4 40]; 
for i = 1:size(S1,2)
    for j = 1:size(S1,1)
        Cl(:,:,i) = Cl(:,:,i) + covar(S1(j,i).L');
        Cr(:,:,i) = Cr(:,:,i) + covar(S1(j,i).R');
    end
end

Cl = Cl ./ size(S1,2);
Cr = Cr ./ size(S1,2);

for i = 1:size(S1,2)
    R(:,:,i) = Cl(:,:,i) + Cr(:,:,i); % (2)
    %% Eigen decomp of R (2)
    [V(:,:,i),D(:,:,i)] = eig(R(:,:,i)); % V = U and D = Sigma as denoted in (2)
    %% Whitening Matrix: P = inverse square root(D) * U' (3)
    P(:,:,i) = sqrtm(inv(D(:,:,i))) * V(:,:,i)';
    %% Sl = P*Cl*P'  Sr = P*Cr*P' (4)
    sL(:,:,i) = P(:,:,i)*Cl(:,:,i)*P(:,:,i)';
    sR(:,:,i) = P(:,:,i)*Cr(:,:,i)*P(:,:,i)';
    
    %% If we eig decomp sL and sR, the sum of eigenvalues should be an
    % identity matrix of NxN. (5)
    [slV2(:,:,i),slD2(:,:,i)] = decompose_S(sL(:,:,i),'ascending');
    [srV2(:,:,i),srD2(:,:,i)] = decompose_S(sR(:,:,i),'descending');
end
check = srD2+slD2; % All are 1, which means I.D Matrix confirmed. (5)
% if check ~= 1
%     error('I.D Matrix not confirmed');
% end

% Projection Matrix: W = U'*P where U' is slV2 and P is whitening matrix (6)
for i = 1:size(S1,2)
    W(:,:,i) = slV2(:,:,i)' * P(:,:,i); %(6)
    % Computing Z: Z = WX (7)
    %    W = W();
    %    Z(:,:,i) =
    %    Z = [];
    %
    m = 2; % First two columns and last 2 columns, like PCA choose most variant columns
    x = W(:,:,i);
    x([3:end-m],:) = [];
    W2(:,:,i) = x;
    for j = 1:size(S1,1)
        S1(j,i).Zl = W2(:,:,i) * S1(j,i).L';
        S1(j,i).Zr = W2(:,:,i) * S1(j,i).R';
    end
    
    % Feature vector array. Trial X Bandwidth: 72 x 36 where 1:4 is 4-8Hz,
    % 5-8 is 8-12Hz and so on.
    for j = 1:size(S1,1)
        var1 = var(S1(j,i).Zl, 1, 2);
        varsum = sum(var1);
        featL(j,[bp(i)-3:bp(i+1)-36]) = log10(var1/varsum);
        clear var1 varsum
        
        var1 = var(S1(j,i).Zr, 1, 2);
        varsum = sum(var1);
        featR(j,[bp(i)-3:bp(i+1)-36]) = log10(var1/varsum);
        clear var1 varsum
    end
    
end

% Adding True label and mergin feature vector
% TRUE LABEL
true_left = zeros(size(featL, 1), 1);
true_right = ones(size(featR, 1), 1);

% FEATURE + TRUE LABEL
left = [featL, true_left];
right = [featR, true_right];

% MERGE LEFT AND RIGHT
train_feat = [left; right];

% SHUFFLE TRAINING DATA
train_feat = train_feat(randperm(size(train_feat, 1)), :);

clearvars -except train_feat S1 bp
%% Mutual Information based selection of Bands
x_train = train_feat(:,[1:end-1]);
y_train = train_feat(:,end);

for i = 1:size(x_train,2)
    % Compute histogram counts for bin and eeg
    N = size(x_train,1); % Number of bins
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
[maxI,Band] = maxk(I,5);
miTrain = train_feat(:,[Band]);

end

function [S1,S2,S3,S4,S5,S6,S7,S8] = preproc(P1,P2,P3,P4,P5,P6,P7,P8)

%% Delete the EOG
P1.EEG(:,[23:25]) = [];
P2.EEG(:,[23:25]) = [];
P3.EEG(:,[23:25]) = [];
P4.EEG(:,[23:25]) = [];
P5.EEG(:,[23:25]) = [];
P6.EEG(:,[23:25]) = [];
P7.EEG(:,[23:25]) = [];
P8.EEG(:,[23:25]) = [];
%% BP each subject 
bp = [4 40];
P1.EEG_filt = bpsubject(P1.EEG,250,bp);
P2.EEG_filt = bpsubject(P2.EEG,250,bp);
P3.EEG_filt = bpsubject(P3.EEG,250,bp);
P4.EEG_filt = bpsubject(P4.EEG,250,bp);
P5.EEG_filt = bpsubject(P5.EEG,250,bp);
P6.EEG_filt = bpsubject(P6.EEG,250,bp);
P7.EEG_filt = bpsubject(P7.EEG,250,bp);
P8.EEG_filt = bpsubject(P8.EEG,250,bp);

%% Left and Right extraction: Each column is the different Filter Band, inside lies L/R data for each trial.

S1 = splitLR(250, P1.Info.EVENT.TYP, P1.Info.EVENT.POS, P1.EEG_filt);
S2 = splitLR(250, P2.Info.EVENT.TYP, P2.Info.EVENT.POS, P2.EEG_filt);
S3 = splitLR(250, P3.Info.EVENT.TYP, P3.Info.EVENT.POS, P3.EEG_filt);
S4 = splitLR(250, P4.Info.EVENT.TYP, P4.Info.EVENT.POS, P4.EEG_filt);
S5 = splitLR(250, P5.Info.EVENT.TYP, P5.Info.EVENT.POS, P5.EEG_filt);
S6 = splitLR(250, P6.Info.EVENT.TYP, P6.Info.EVENT.POS, P6.EEG_filt);
S7 = splitLR(250, P7.Info.EVENT.TYP, P7.Info.EVENT.POS, P7.EEG_filt);
S8 = splitLR(250, P8.Info.EVENT.TYP, P8.Info.EVENT.POS, P8.EEG_filt);

clear P1 P2 P3 P4 P5 P6 P7 P8 eegL eegR 

end

function y = bpsubject(subjectdata,fs,bp)
x = subjectdata;

for i = 1:length(bp)-1
    tic
    y(:,:,i) = bandpass(x,[bp(i) bp(i+1)],fs);
    toc
end
end

function S = splitLR(fs, typ, pos, eeg)
tMI = 3 * fs; % Number of samples in 3 seconds of MI.

typ == 769; % L
idxL = ans;
typ == 770; % R
idxR = ans;

locL = pos(idxL);
locR = pos(idxR);

clear idxR idxL ans

for i = 1:size(eeg,3)
    for j = 1:length(locL)
        eegL = eeg([locL(j):locL(j)+tMI],:,i);
        eegR = eeg([locR(j):locR(j)+tMI],:,i);
        S(j,i) = struct('L',eegL,'R',eegR);
    end
end

end

function c = covar(x) % Equation (1) [1]
    A = x * x';
    t = trace(A);
    c = A/t;
end

function [B, lambda] = decompose_S(S_one_class, order)
    % This function decomposes the S matrix of one class to get the eigenvector.
    % Both eigenvectors will be the same but in opposite order.
    % i.e., the highest eigenvector in S left will be equal to the lowest eigenvector in S right matrix.
    %
    % Parameters:
    % - S_one_class: The S matrix of one class
    % - order: Order in which to sort eigenvalues and eigenvectors ('ascending' or 'descending')
    %
    % Returns:
    % - B: Eigenvectors sorted according to eigenvalues
    % - lambda: Eigenvalues sorted according to the specified order

    [B, lambda] = eig(S_one_class);
    lambda = diag(lambda);
    
    if strcmp(order, 'ascending')
        [~, idx] = sort(lambda, 'ascend');
    elseif strcmp(order, 'descending')
        [~, idx] = sort(lambda, 'descend');
    else
        error('Wrong order input');
    end
    
    lambda = lambda(idx);
    B = B(:, idx);
end

function H = entropy(px)
    idx = px > 0;
    H = -sum(px(idx) .* log2(px(idx))); % This just takes the entropy formula and gets rid of the px=0 terms
end

function I = MInformation(spikeTrain, N, phi_bin)

    flatphi = phi_bin(:);
    spikeTrain = spikeTrain(:);

    cPhi = histcounts(flatphi, N); % Convert data into probabilities
    cSpike = histcounts(spikeTrain, N);
    cPS = histcounts2(flatphi, spikeTrain, N);
    Pp = cPhi / sum(cPhi);
    Ps = cSpike / sum(cSpike);
    Pps = cPS / sum(cPS(:));

    Hp = entropy(Pp);
    Hs = entropy(Ps);
    Hps = entropy(Pps);
    I = Hp + Hs - Hps;
end