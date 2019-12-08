clc;close all;
%% Train

for k = 3:length(u)
     
    x(k-2,:) = [y(k-1,1) y(k-2,1) u(k) u(k-1)];
    
    q(k-2,:) = y(k,:) ;
    
end

nTrain= 0.7*length(u);
TrainInputs  = x(1:nTrain,:);
TrainTargets = q(1:nTrain,:);

TestInputs  = x(nTrain:end,:);
TestTargets = q(nTrain:end,:);

net =feedforwardnet([15]);
net.performFcn = 'mse';
net.trainParam.max_fail=10;
net.trainParam.epochs = 550;
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