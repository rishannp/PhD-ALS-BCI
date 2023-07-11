function [vectors, dictionary] = pcfcn(X,sdp)
%X = fr_t(:,:,2)'; % Features = columns and Measures = rows
%Xmean=mean(X); % Cov requires mean centered
%B=X-Xmean;
X = X';
C = cov(X); % Cov will mean center it anyways
[vectors, values] = eig(C); % Eigendecomposition
[~,col]=maxk(diag(values),3); % find something ~95% variance
represent = values(col,col)/sum(diag(values));
represent = diag(represent);
represents = sum(represent); % explained var
vectors = vectors(:,col);
sd = std(vectors);
sd = sdp.*sd; % Find 2 sd increase and class it as statistically significant in order to gauge maximal
% contribution
dictionary(:,1) = vectors(:,1) >= sd(1);% + sd(1);
% dictionary(:,2) = vectors(:,2) >= sd(2);
% dictionary(:,3) = vectors(:,3) >= sd(3);
dictionary = double(dictionary);

% x1 = 100*represent(1);
% txt1 = ['Variance explained : ',num2str(x1)];
% 
% x2 = 100*represent(2);
% txt2 = ['Variance explained : ',num2str(x2)];
% 
% x3 = 100*represent(3);
% txt3 = ['Variance explained : ',num2str(x3)];
% 
% figure()
% subplot(3,1,1)
% scatter([1:size(X,2)],vectors(:,1));
% yline(sd(1),'--r');
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