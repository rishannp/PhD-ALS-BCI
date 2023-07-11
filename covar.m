function c = covar(x) % Equation (1) [1]
    A = x * x';
    t = trace(A);
    c = A/t;
end
