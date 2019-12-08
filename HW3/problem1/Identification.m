clc;close all;
%% Train

for k = 3:length(u)
     
    x(k-2,:) = [ u(k) u(k-1) u(k-2) y(k-1,1) y(k-2,1)];
    
    q(k-2,:) = y(k,:) ;
    
end

TrainInputs  = x(1:700,:);
TrainTargets = q(1:700,:);

TestInputs  = x(701:end,:);
TestTargets = q(701:end,:);

net =feedforwardnet([5]);
net.performFcn = 'mse';
net.trainParam.max_fail=10;
net.trainParam.epochs = 250;
net = train(net,TrainInputs',TrainTargets');
TrainOutputs = net(TrainInputs');
TestOutputs = net(TestInputs');

%%
figure;
plot(TrainTargets,'-or','Linewidth',2);
hold on
grid on
plot(TrainOutputs','-*b','Linewidth',0.5);
xlabel('sample')
ylabel('V')
axis('square')
legend('Target', 'Model')
title('Train Data')
%%
figure;
plot(TestTargets,'-or','Linewidth',2);
hold on
grid on
plot(TestOutputs','-*b','Linewidth',0.5);
xlabel('sample')
ylabel('V')
axis('square')
legend('Target', 'Model')
title('Test Data')