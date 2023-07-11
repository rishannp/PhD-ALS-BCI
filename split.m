function S = split(fs, target, eeg)
%tMI = 5 * fs; % Number of samples in 5 seconds of MI.

x = target; 
x == 1
L = ans;
L(end) = 0;
clear ans

x == 2 
R = ans; 
R(end) = 0;
clear ans

x == 0
Re = ans;
Re(1) = 0;
Re(end) = 0;
clear ans

flagL = zeros(1,length(x));
flagR = zeros(1,length(x));
flagRe = zeros(1,length(x));

for i = 1:length(L)
    % find first 1
    if L(i) == 1 && L(i-1) == 0 && L(i+1) == 1
        flagL(i) = 1;
    end
    if L(i) == 1 && L(i-1) == 1 && L(i+1) == 0
        flagL(i) = 1;
    end
    
    if R(i) == 1 && R(i-1) == 0 && R(i+1) == 1
        flagR(i) = 1;
    end
    if R(i) == 1 && R(i-1) == 1 && R(i+1) == 0
        flagR(i) = 1;
    end
    
    if Re(i) == 1 && Re(i-1) == 0 && Re(i+1) == 1
        flagRe(i) = 1;
    end
    if Re(i) == 1 && Re(i-1) == 1 && Re(i+1) == 0
        flagRe(i) = 1;
    end
end

[~,locL] = findpeaks(flagL);
[~,locR] = findpeaks(flagR);
[~,locRe] = findpeaks(flagRe);

clear flagL flagR flagRe L R Re x 

for i = 1:9
    for j = 1:2:(length(locL))-1
        eegL = eeg([locL(j):locL(j+1)],:,i); 
        eegR = eeg([locR(j):locR(j+1)],:,i);
        eegRe = eeg([locRe(j):locRe(j+1)],:,i);
        S(j,i) = struct('L',eegL,'R',eegR,'Re',eegRe);
    end
end
S([2:2:size(S,1)],:) = [];
end