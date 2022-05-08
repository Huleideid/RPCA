function [new_m,new_avg,new_var,new_cor,stdX] = RPCA(old_m,old_avg,old_var,old_cor,newX,f)  
    [m,n]=size(newX);
    new_m=old_m+m;
     if nargin < 6
        f=old_m/new_m;%f  
     end
%      Update the mean values 
%     new_avg=1/(new_m)*(old_m*old_avg+newX'*ones(m,1));
    new_avg=f*old_avg+(1-f)/m*newX'*ones(m,1);   
    delta_avg=new_avg-old_avg;
    new_var=zeros(n,1);
%     for i=1:n
%         new_var(i)=1/(new_m-1)*((old_m-1)*old_var(i)+old_m*delta_avg(i)^2+norm(newX(:,i)-new_avg(i)*ones(m,1))^2);
%     end
%      Update standard deviation
    for i=1:n
        new_var(i)=f*(old_var(i)+delta_avg(i)^2)+(1-f)/m*norm(newX(:,i)-new_avg(i)*ones(m,1))^2;
    end
    stdX=(newX-repmat(new_avg',m,1))*(diag(new_var')^(-1/2));
%     new_cor=(old_m-1)/((new_m-1))*pinv(diag(new_var)^(1/2))*(diag(old_var)^(1/2)*old_cor*diag(old_var)^(1/2))*pinv(diag(new_var)^(1/2))+old_m/((new_m-1))*pinv(diag(new_var)^(1/2))*delta_avg*delta_avg'*pinv(diag(new_var)^(1/2))+1/((new_m-1))*stdX'*stdX;
    new_cor=f*pinv(diag(new_var)^(1/2))*(diag(old_var)^(1/2)*old_cor*diag(old_var)^(1/2)+delta_avg*delta_avg')*pinv(diag(new_var)^(1/2))+(1-f)/m*stdX'*stdX;
end