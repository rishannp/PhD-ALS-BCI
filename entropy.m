function H = entropy(px)
    idx = px > 0;
    H = -sum(px(idx) .* log2(px(idx))); % This just takes the entropy formula and gets rid of the px=0 terms
end