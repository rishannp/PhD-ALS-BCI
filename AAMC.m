function [trainacc, testacc, mean1] = AAMC(t,winlen,S9,window)

x = 0:t:size(S9,1);
x(end) = length(S9); % Data splits (5)
accuracy = [0;0;0];

for i = 1:length(x)-2
    
    if i == 1
        ltr = S9([x(i)+1:x(i+1)],:);
        rtr = S9([x(i)+1:x(i+1)],:);
        retr = S9([x(i)+1:x(i+1)],:);
        
        [reBand,Wre,reTrain,reLabel] = fbcspRe(ltr,2); % Training data using ltr bc the functions draws the right matrices
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr,2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(ltr,2);
        
        % Accuracy var is L,R,Re in order of rows.
        id = size(accuracy,2); % update accuracy end
        [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % 5 Fold CV
        [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
        [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
        
        test = S9([x(i+1)+1:x(i+2)],:); % next piece of testing data
        
        [l.test,l.label] = Ltest1(test,Wl,2,lBand); % finding feature vector of l/r/re
        [re.test,re.label] = Retest1(test,Wre,2,reBand);
        [r.test,r.label] = Rtest1(test,Wr,2,rBand);
        
        out.l = mdl.L.predictFcn(l.test);  % make predictions based on previous trained model
        out.r = mdl.R.predictFcn(r.test);
        out.re = mdl.Re.predictFcn(re.test);
        
        id = size(accuracy,2); % update accuracy end
        l.label == out.l; %% Test
        accuracy(1,id+1) = (sum(ans)/length(ans)) ;
        correct.l = ans;
        
        r.label == out.r;
        accuracy(2,id+1) = (sum(ans)/length(ans)) ;
        correct.r = ans;
        
        re.label == out.re;
        accuracy(3,id+1) = (sum(ans)/length(ans)) ;
        correct.re = ans;
        
        idxl = [];
        idxr = [];
        idxre = [];
        for j = 1:length(correct.l)/2
            if correct.l(j) == 1
                %lTrain = vertcat(lTrain,l.test(j,:));
                %lLabel = vertcat(lLabel,l.label(j));
                idxl = horzcat(idxl,x(i)+j);
                if idxl < 5
                    ltr = vertcat(ltr,test(j,:));
                end
                if j > 5
                    ltr = vertcat(ltr,test([j-5],:));
                end
            end
            
            if correct.r(j) == 1
                %rTrain = vertcat(rTrain,r.test(j,:));
                %rLabel = vertcat(rLabel,r.label(j));
                idxr = horzcat(idxr,x(i)+j);
                if j < 5
                    rtr = vertcat(rtr,test(j,:));
                end
                if j > 5
                    rtr = vertcat(rtr,test([j-5],:));
                end
                
            end
            
            if correct.re(j) == 1
                %reTrain = vertcat(reTrain,re.test(j,:));
                %reLabel = vertcat(reLabel,re.label(j));
                idxre = horzcat(idxre,x(i)+j);
                if j < 5
                    retr = vertcat(retr,test(j,:));
                end
                if j > 5
                    retr = vertcat(retr,test([j-5],:));
                end
            end
        end
        
        
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr,2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr,2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr,2);
        
        id = size(accuracy,2); % update accuracy end
        [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % Retrain classifier
        [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
        [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
        
        %         figure()
        %         subplot(3,1,1)
        %         gscatter(lTrain(:,1),lTrain(:,2),lLabel);
        %         subplot(3,1,2)
        %         gscatter(rTrain(:,1),rTrain(:,2),rLabel);
        %         subplot(3,1,3)
        %         gscatter(reTrain(:,1),reTrain(:,2),reLabel);
        
    end
    
    test = S9([x(i+1)+1:x(i+2)],:); % next piece of testing data
    
    [l.test,l.label] = Ltest1(test,Wl,2,lBand); % finding feature vector of l/r/re
    [re.test,re.label] = Retest1(test,Wre,2,reBand);
    [r.test,r.label] = Rtest1(test,Wr,2,rBand);
    
    %     figure()
    %     subplot(3,1,1)
    %     gscatter(l.test(:,1),l.test(:,2),l.label);
    %     subplot(3,1,2)
    %     gscatter(r.test(:,1),r.test(:,2),r.label);
    %     subplot(3,1,3)
    %     gscatter(re.test(:,1),re.test(:,2),re.label);
    
    out.l = mdl.L.predictFcn(l.test);  % make predictions based on previous trained model
    out.r = mdl.R.predictFcn(r.test);
    out.re = mdl.Re.predictFcn(re.test);
    
    id = size(accuracy,2); % update accuracy end
    l.label == out.l; %% Test
    accuracy(1,id+1) = (sum(ans)/length(ans)) ;
    correct.l = ans;
    
    r.label == out.r;
    accuracy(2,id+1) = (sum(ans)/length(ans)) ;
    correct.r = ans;
    
    re.label == out.re;
    accuracy(3,id+1) = (sum(ans)/length(ans)) ;
    correct.re = ans;
    
    idxl = [];
    idxr = [];
    idxre = [];
    for j = 1:length(correct.l)/2
        if correct.l(j) == 1
            %lTrain = vertcat(lTrain,l.test(j,:));
            %lLabel = vertcat(lLabel,l.label(j));
            idxl = horzcat(idxl,x(i)+j);
            if idxl < 5
                ltr = vertcat(ltr,test(j,:));
            end
            if j > 5
                ltr = vertcat(ltr,test([j-5],:));
            end
        end
        
        if correct.r(j) == 1
            %rTrain = vertcat(rTrain,r.test(j,:));
            %rLabel = vertcat(rLabel,r.label(j));
            idxr = horzcat(idxr,x(i)+j);
            if j < 5
                rtr = vertcat(rtr,test(j,:));
            end
            if j > 5
                rtr = vertcat(rtr,test([j-5],:));
            end
            
        end
        
        if correct.re(j) == 1
            %reTrain = vertcat(reTrain,re.test(j,:));
            %reLabel = vertcat(reLabel,re.label(j));
            idxre = horzcat(idxre,x(i)+j);
            if j < 5
                retr = vertcat(retr,test(j,:));
            end
            if j > 5
                retr = vertcat(retr,test([j-5],:));
            end
        end
    end
    
    if window == 1
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr([end-winlen:end],:),2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr([end-winlen:end],:),2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr([end-winlen:end],:),2);
    elseif window == 0
        [reBand,Wre,reTrain,reLabel] = fbcspRe(retr,2); % Retrain the spatial filter with the additional correct trials from testing
        [lBand,Wl,lTrain,lLabel] = fbcspL(ltr,2);
        [rBand,Wr,rTrain,rLabel] = fbcspR(rtr,2);
    end
    
    %     figure()
    %     subplot(3,1,1)
    %     gscatter(lTrain(:,1),lTrain(:,2),lLabel);
    %     subplot(3,1,2)
    %     gscatter(rTrain(:,1),rTrain(:,2),rLabel);
    %     subplot(3,1,3)
    %     gscatter(reTrain(:,1),reTrain(:,2),reLabel);
    
    id = size(accuracy,2); % update accuracy end
    [mdl.L, accuracy(1,id+1)] = trainQDA(lTrain, lLabel); % Retrain classifier
    [mdl.R, accuracy(2,id+1)] = trainQDA(rTrain, rLabel);
    [mdl.Re, accuracy(3,id+1)] = trainQDA(reTrain, reLabel);
end

%
% Find columns with all zeros
zero_columns = all(accuracy == 0);

% Delete columns from the matrix
accuracy(:, zero_columns) = [];

testacc = accuracy(:,[2:2:length(accuracy)]);
trainacc = accuracy(:,[1:2:length(accuracy)]);

[mean1] = fig(trainacc, testacc);

end