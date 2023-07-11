function [B, lambda] = decompose_S(S_one_class, order)
    % This function decomposes the S matrix of one class to get the eigenvector.
    % Both eigenvectors will be the same but in opposite order.
    % i.e., the highest eigenvector in S left will be equal to the lowest eigenvector in S right matrix.
    %
    % Parameters:
    % - S_one_class: The S matrix of one class
    % - order: Order in which to sort eigenvalues and eigenvectors ('ascending' or 'descending')
    %
    % Returns:
    % - B: Eigenvectors sorted according to eigenvalues
    % - lambda: Eigenvalues sorted according to the specified order

    [B, lambda] = eig(S_one_class);
    lambda = diag(lambda);
    
    if strcmp(order, 'ascending')
        [~, idx] = sort(lambda, 'ascend');
    elseif strcmp(order, 'descending')
        [~, idx] = sort(lambda, 'descend');
    else
        error('Wrong order input');
    end
    
    lambda = lambda(idx);
    B = B(:, idx);
end
