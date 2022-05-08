function [V,D,pcnumber] = pc_number(X)
% Calculate the PC number that captures 87% variance
[V, D] = svd(X);
% [D,Index]=sort(diag(D)','descend');
% D=diag(D);
% V=V(:,Index);
% D=diag(D);
% D=fliplr(D');
% D=diag(D);
% permute
if size(D,2)== 1
    pcnumber = 1;
else
    S = diag(D);
    S = S(diag(D)>0);
    i = 0;
    var = 0;
    while var <.85*sum(S)       
        i = i+1;
        var = var+S(i);
    end
    pcnumber = i;
end
