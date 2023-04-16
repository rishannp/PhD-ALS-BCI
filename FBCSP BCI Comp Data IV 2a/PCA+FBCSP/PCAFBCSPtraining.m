%% PCA
clear all
clc

load trainingData.mat
%%
t0 = tic();


t1 = tic();
[vectors, dictionary] = pcfcn(P1.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P1.EEG = P1.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P2.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P2.EEG = P2.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P3.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P3.EEG = P3.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P4.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P4.EEG = P4.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P5.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P5.EEG = P5.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P6.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P6.EEG = P6.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P7.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P7.EEG = P7.EEG(:,[dictionary]);

[vectors, dictionary] = pcfcn(P8.EEG(:,[1:22])');
dictionary = [1:22] .* dictionary';
dictionary = dictionary(dictionary~=0);
P8.EEG = P8.EEG(:,[dictionary]);

pctime = toc(t1);

clear vectors dictionary
%%
save('PCAtrainingData');

%% BPF 
t3 = tic();
[S1,S2,S3,S4,S5,S6,S7,S8] = preproc(P1,P2,P3,P4,P5,P6,P7,P8);
bpftime = toc(t3);

clear P1 P2 P3 P4 P5 P6 P7 P8

%% train
s = size(S1,1)*0.75;
s = round(s);
t4 = tic();
[x_train1,y_train1] = fbcsp(S1([1:s],:));
[x_train2,y_train2] = fbcsp(S2([1:s],:));
[x_train3,y_train3] = fbcsp(S3([1:s],:));
[x_train4,y_train4] = fbcsp(S4([1:s],:));
[x_train5,y_train5] = fbcsp(S5([1:s],:));
[x_train6,y_train6] = fbcsp(S6([1:s],:));
[x_train7,y_train7] = fbcsp(S7([1:s],:));
[x_train8,y_train8] = fbcsp(S8([1:s],:));



save('PCAxyTrain');

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
save('PCAxyTest');
fbcsptime = toc(t4);

totaltime = toc(t0);

%% Classifier
Mdl1 = fitcknn(x_train1,y_train1,'NumNeighbors',5,'Standardize',1);
y1out = predict(Mdl1,x_test1);

accuracy1 = sum((y1out == y_test1)/length(y1out)) * 100;

Mdl2 = fitcknn(x_train2,y_train2,'NumNeighbors',5,'Standardize',1);
y2out = predict(Mdl2,x_test2);

accuracy2 = sum((y2out == y_test2)/length(y2out)) * 100;


Mdl3 = fitcknn(x_train3,y_train3,'NumNeighbors',5,'Standardize',1);
y3out = predict(Mdl3,x_test3);

accuracy3 = sum((y3out == y_test3)/length(y3out)) * 100;


Mdl4 = fitcknn(x_train4,y_train4,'NumNeighbors',5,'Standardize',1);
y4out = predict(Mdl4,x_test4);

accuracy4 = sum((y4out == y_test4)/length(y4out)) * 100;

Mdl5 = fitcknn(x_train5,y_train5,'NumNeighbors',5,'Standardize',1);
y5out = predict(Mdl5,x_test5);

accuracy5 = sum((y5out == y_test5)/length(y5out)) * 100;

Mdl6 = fitcknn(x_train6,y_train6,'NumNeighbors',5,'Standardize',1);
y6out = predict(Mdl6,x_test6);

accuracy6 = sum((y6out == y_test6)/length(y6out)) * 100;


Mdl7 = fitcknn(x_train7,y_train7,'NumNeighbors',5,'Standardize',1);
y7out = predict(Mdl7,x_test7);

accuracy7 = sum((y7out == y_test7)/length(y7out)) * 100;


Mdl8 = fitcknn(x_train8,y_train8,'NumNeighbors',5,'Standardize',1);
y8out = predict(Mdl8,x_test8);

accuracy8 = sum((y8out == y_test8)/length(y8out)) * 100;

%%
function [vectors, dictionary] = pcfcn(X)
%X = fr_t(:,:,2)'; % Features = columns and Measures = rows
%Xmean=mean(X); % Cov requires mean centered
%B=X-Xmean;
X = X';
C = cov(X); % Cov will mean center it anyways 
[vectors, values] = eig(C); % Eigendecomposition 
[~,col]=maxk(diag(values),1); % find something ~95% variance
represent = values(col,col)/sum(diag(values));
represent = diag(represent);
represents = sum(represent); % explained var
vectors = vectors(:,col); 
sd = std(vectors);
sd = 1.*sd; % Find 2 sd increase and class it as statistically significant in order to gauge maximal
% contribution
dictionary(:,1) = vectors(:,1) >= mean(vectors);% + sd(1);
% dictionary(:,2) = vectors(:,2) >= sd(2);
% dictionary(:,3) = vectors(:,3) >= sd(3);
dictionary = double(dictionary);

x1 = 100*represent(1);
txt1 = ['Variance explained : ',num2str(x1)];

% x2 = 100*represent(2);
% txt2 = ['Variance explained : ',num2str(x2)];
% 
% x3 = 100*represent(3);
% txt3 = ['Variance explained : ',num2str(x3)];

% figure()
% %subplot(3,1,1)
% scatter([1:size(X,2)],vectors(:,1)); 
% yline(mean(vectors),'--r');
% xlabel('Neuron')
% ylabel('Variance')
% title(txt1)
% 
% subplot(3,1,2) 
% scatter([1:size(X,2)],vectors(:,2)); 
% yline(sd(2),'--r');
% xlabel('Neuron')
% ylabel('Variance')
% title(txt2)
% 
% subplot(3,1,3) 
% scatter([1:size(X,2)],vectors(:,3));
% yline(sd(3),'--r');
% xlabel('Neuron')
% ylabel('Variance')
% title(txt3)


%PC = transpose(vectors) .* X; % Bringing back to OG data space
  end

function [miTrain,y_train] = fbcsp(S1) % Following paper "Common Spatial Pattern Method for Channel Selection in
% Motor Imagery Based Brain-Computer Interface" by Wang et al - [1]
%% Average covariance of Left and Right
Cl = zeros(size(S1(1,1).L,2),size(S1(1,1).L,2),size(S1,2));
Cr = zeros(size(S1(1,1).L,2),size(S1(1,1).L,2),size(S1,2));
featL = zeros(size(S1,1),36);
featR = zeros(size(S1,1),36);
bp = 4:4:40; 
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
        featL(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum
        
        var1 = var(S1(j,i).Zr, 1, 2);
        varsum = sum(var1);
        featR(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
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

for i = 1:36
    % Compute histogram counts for bin and eeg
    N = 144; % Number of bins
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

% BP each subject 
bp = 4:4:40;
%4-8Hz 8-12Hz 12-16Hz 16-20Hz 20-24Hz 24-28Hz 28-32Hz 32-36Hz 36-40Hz

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
    
    y(:,:,i) = bandpass(x,[bp(i) bp(i+1)],fs);
    
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

for i = 1:9
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