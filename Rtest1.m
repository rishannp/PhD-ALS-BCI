function [train,label] = Rtest1(S1,W,m,Band)
bp = 4:4:40;
for i = 1:size(S1,2)
    %m = 2; % First two columns and last 2 columns, like PCA choose most variant columns
    x = W(:,:,i);
    x([m+1:end-m],:) = [];
    W2(:,:,i) = x;
    for j = 1:size(S1,1)
        S1(j,i).Zr = W2(:,:,i) * S1(j,i).R';
        S1(j,i).Zrre = W2(:,:,i) * S1(j,i).LRE';
    end
    
    % Feature vector array. Trial X Bandwidth: 72 x 36 where 1:4 is 4-8Hz,
    % 5-8 is 8-12Hz and so on.
    for j = 1:size(S1,1)
        var1 = var(S1(j,i).Zr, 1, 2);
        varsum = sum(var1);
        featR(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum

        var1 = var(S1(j,i).Zrre, 1, 2);
        varsum = sum(var1);
        featRre(j,[bp(i)-3:bp(i+1)-4]) = log10(var1/varsum);
        clear var1 varsum
    end
    
end

true_right = ones(size(featR, 1), 1);
true_rest = zeros(size(featRre,1),1);

% FEATURE + TRUE LABEL
right = [featR, true_right];
rest = [featRre, true_rest];

% MERGE

train_feat = [right; rest];

miTrain = train_feat(:,[Band]);
miTrain = [miTrain,train_feat(:,end)];
train = miTrain(:,[1:end-1]);
train = double(train);
label = miTrain(:,end);
end