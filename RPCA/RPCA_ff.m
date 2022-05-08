function [x,new_avg,new_var,new_cor,f,nt] = RPCA_ff(xt,old_avg,old_var,old_cor,f,nt)
    [m,n]=size(xt);
%% Step 1. Application of EWPCA to a new observation
%     x=(xt-old_avg')./(old_var'.^(1/2));
%     new_cor=f*old_cor+(1-f)*x'*x;
%     [V,D,pcnumber] = pc_number(new_cor);
%     P=V(1,:pcnumber);
%% Step 3. Update the EWPCA parameter
%   Update the mean values 
%     new_avg=f*old_avg;
    new_avg=f*old_avg+(1-f)/m*xt'*ones(m,1); 
%   Update the data matrix
%     Xt=f*[X;x];
%   Update standard deviation  
%     new_var=f*old_var;
    delta_avg=new_avg-old_avg;
    new_var=zeros(n,1);
    for i=1:n
        new_var(i)=f*(old_var(i)+delta_avg(i)^2)+(1-f)/m*norm(xt(:,i)-new_avg(i)*ones(m,1))^2;
    end
%     new_cor=f*old_cor+(1-f)*x'*x;
    x=(xt-repmat(new_avg',m,1))*(diag(new_var')^(-1/2));
    new_cor=f*pinv(diag(new_var)^(1/2))*(diag(old_var)^(1/2)*old_cor*diag(old_var)^(1/2)+delta_avg*delta_avg')*pinv(diag(new_var)^(1/2))+(1-f)/m*(x')*x;
    [V,D,pcnumber] = pc_number(new_cor);
    lambda= D(1:pcnumber,1:pcnumber);
    P=V(:,1:pcnumber);
%   Update the exponential weighting and the asymptotic memory length,
%     f=1-(1-x*(P)*P'*x')/n*(x*(eye(n)-(P)*P')*x'/n)/(sqrt(nt));
    f=1-(1-x*(P*P')*x')/n*(x*(eye(n)-(P*P'))*x'/n)/((nt));
    nt=min(floor(1/(1-f)),100);
end
