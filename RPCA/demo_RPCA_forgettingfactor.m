% function demo()
clear
clc
close all
A=[0.3723 0.6815;
    0.4890 0.2954;
    0.9842 0.1793];
%% 训练数据
model_number=300;
x_train=[];
for i=1:model_number
    s1=normrnd(10,0.8);
    s2=normrnd(12,1.3);
    ek=normrnd(0,0.2^2,3,1);
    sk=[s1;s2];
    if i>100
        ek=ek+0.2*(i-100);
    end
    xk =A*sk+ek;
    x_train=[x_train;xk'];
end


X=x_train(1:100,:);
testX=x_train(101:300,:);

% X=ones(3);
% testX=3*[1,2,2;2,4,6];

old_m=size(X,1);
avg=mean(X)';
var=std(X)';
cor=cov(zscore(X));
%% Step 1. Initialization
f=1-1/old_m;
nt=old_m;
%% Step 2,3
T2=[];
SPE=[];
T2UCL=[];
SPEUCL=[];
factor=[];
% nt=[];
for i=1:size(testX,1)
    xt=testX(i,:);
%     [old_m,old_avg,old_var,old_cor,stdX] = RPCA(old_m,old_avg,old_var,old_cor,newX,f);
    [x,avg,var,cor,f,nt]=RPCA_ff(xt,avg,var,cor,f,nt);
    [m,n]=size(x);
    [V,D,num_pc] = pc_number(cor);
    P=V(:,1:num_pc);
    lambda=D(1:num_pc,1:num_pc);
    for j=1:m
        spe(j)=x(j,:)*(eye(n)-P*P')*x(j,:)';
        t2(j)=x(j,:)*P*pinv(lambda)*P'*x(j,:)';
        %求置信度为90
        %、95%时的T2统计控制限                       
        t2UCL(j)=chi2inv(0.9, num_pc);
        %置信度为95%的Q统计控制限
        for a = 1:3
            theta(a) = sum(diag(D(num_pc+1:n,num_pc+1:n)).^a);
        end
        h0 = 1 - 2*theta(1)*theta(3)/(3*theta(2)^2);
        ca = norminv(0.95,0,1);
        speUCL(j) = theta(1)*(h0*ca*sqrt(2*theta(2))/theta(1) + 1 + theta(2)*h0*(h0 - 1)/theta(1)^2)^(1/h0);   
    end
    T2=[T2 t2];
    SPE=[SPE spe];
    T2UCL=[T2UCL t2UCL];
    SPEUCL=[SPEUCL speUCL];
    factor=[factor f];
end
%% 
figure
subplot(2,1,1)
plot(T2)
hold on
plot(T2UCL)
subplot(2,1,2)
plot(SPE)
hold on
plot(SPEUCL)
%% 
figure
plot(factor)
