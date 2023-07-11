load Subject1.mat

for i = 5:1:35
    [trainacc, testacc, mean1] = AAMC(i,winlen,S9,0);
    y1(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S9

load Subject2.mat

for i = 5:1:35
    [trainacc, testacc, mean1] = AAMC(i,winlen,S21,0);
    y2(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S21

load Subject3.mat

for i = 5:1:35
    [trainacc, testacc, mean1] = AAMC(i,winlen,S34,0);
    y3(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S34

%% plots

figure()
subplot(3,1,1)
plot(y1(1,:)); hold on; plot(y1(2,:)); hold on; plot(y1(3,:));
yline(0.7,'k');
title('S1');
ylabel('Accuracy');
xlabel('Epoch Number');

subplot(3,1,2)
plot(y2(1,:)); hold on; plot(y2(2,:)); hold on; plot(y2(3,:));
yline(0.7,'k');
title('S2');
ylabel('Accuracy');
xlabel('Epoch Number');

subplot(3,1,3)
plot(y3(1,:)); hold on; plot(y3(2,:)); hold on; plot(y3(3,:));
yline(0.7,'k');
title('S3');
ylabel('Accuracy');
xlabel('Epoch Number');

y11 = mean(y1);
y22 = mean(y2);
y33 = mean(y3); 

figure()
plot(y11,'r');
hold on; 
plot(y22,'g');
hold on; 
plot(y33,'b');
yline(0.7,'k');
legend('S1','S2','S3')
title('Average accuracy with epoch');
ylabel('Accuracy');
xlabel('Epoch Number');

%% 
load Subject1.mat
[S1trainacc, S1testacc, S1mean1] = AAMC(11,0,S9,0);


load Subject2.mat
[S2trainacc, S2testacc, S2mean1] = AAMC(12,0,S21,0);


load Subject3.mat
[S3trainacc, S3testacc, S3mean1] = AAMC(15,0,S34,0);

%%

load Subject1.mat

for i = 5:1:15
    [trainacc, testacc, mean1] = AAMC(11,i,S9,1);
    y1(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S9
%%
load Subject1.mat
[trainacc, testacc, mean1] = AAMC(11,20,S9,1);
%%
load Subject2.mat

for i = 5:1:15
    [trainacc, testacc, mean1] = AAMC(12,i,S21,1);
    y2(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S21

load Subject3.mat

for i = 5:1:15
    [trainacc, testacc, mean1] = AAMC(15,i,S34,1);
    y3(:,i) = mean1.test; 
    clear mean1 test acc trainacc
end

clear S34
