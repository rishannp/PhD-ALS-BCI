function S = LRvRe(S)
    for i = 1:size(S,1)
        for j = 1:size(S,2)
            S(i,j).LR = vertcat(S(i,j).L,S(i,j).R); 
            S(i,j).RRE = vertcat(S(i,j).R,S(i,j).Re); 
            S(i,j).LRE = vertcat(S(i,j).L,S(i,j).Re); 
        end
    end
end