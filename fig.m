function [mean1] = fig(trainacc,testacc)

% figure()
% subplot(1,3,1)
% bar(trainacc(1,:),'k');
% yline(mean(trainacc(1,:)),'r');
% yline(0.7,'g');
% title('Left');
% xlabel('Epoch');
% ylabel('Accuracy');
% ylim([0 1]);
% subplot(1,3,2)
% bar(trainacc(2,:),'k');
% yline(mean(trainacc(2,:)),'r');
% yline(0.7,'g');
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Right');
% ylim([0 1]);
% subplot(1,3,3)
% bar(trainacc(3,:),'k');
% yline(mean(trainacc(3,:)),'r');
% yline(0.7,'g');
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Rest');
% ylim([0 1]);
% sgtitle('Training Accuracies');

figure()
subplot(1,3,1)
bar(testacc(1,:),'k');
yline(mean(testacc(1,:)),'r');
yline(0.7,'g');yline(0.7,'g');
xlabel('Test Number');
ylabel('Accuracy');
title('Left');
ylim([0 1]);
subplot(1,3,2)
bar(testacc(2,:),'k');
yline(mean(testacc(2,:)),'r');
yline(0.7,'g');
xlabel('Test Number');
ylabel('Accuracy');
title('Right');
ylim([0 1]);
subplot(1,3,3)
bar(testacc(3,:),'k');
yline(mean(testacc(3,:)),'r');
yline(0.7,'g');
xlabel('Test Number');
ylabel('Accuracy');
title('Rest');
ylim([0 1]);
sgtitle('Testing Accuracies');

mean1.train = mean(trainacc,2);
mean1.test = mean(testacc,2);

% testflag = 0;
% for i = 1:length(testacc)
%     if testacc(i) < 0.70 
%         testflag = testflag + 1; 
%     else
%         testflag = testflag;
%     end
% end
% 
% trainflag = 0;
% for i = 1:length(trainacc)
%     if trainacc(i) < 0.70 
%         trainflag = trainflag + 1; 
%     else
%         trainflag = trainflag;
%     end
% end

% display(trainflag);
display(mean1.train);
% display(testflag);
display(mean1.test);
end

