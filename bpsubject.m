function y = bpsubject(subjectdata,fs,bp)
x = subjectdata;

for i = 1:length(bp)-1
    
    y(:,:,i) = bandpass(x,[bp(i) bp(i+1)],fs);
    
end
end