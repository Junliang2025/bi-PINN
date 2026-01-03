clc
clear
close all
format long
rand('seed',3);

%% 边界条件
numB = 21;
numBoundary = [numB numB];

theta0BC1 = linspace(0,2*pi,numBoundary(1));
r0BC1 = 1-0.2.*cos(5.*theta0BC1);
BC1 = zeros(1,numBoundary(1));

theta0BC2 = linspace(0,2*pi,numBoundary(2));
r0BC2 = linspace(0,0,numBoundary(2));
BC21 = -1.*ones(1,numBoundary(2))+0.2.*cos(5.*theta0BC2);
BC2t = -sin(5.*theta0BC2);
%% 内部 
numIn = 100;
x1 = linspace(-1,1,numIn);
data1 = [];
for i=1:numIn
    x2 = repmat(x1(i),1,numIn);
    y2 = linspace(-1,1,numIn);
    data2 = [x2;y2];
    data1 = cat(2,data1,data2);
end

dataX = data1(1,:); 
dataY = data1(2,:);
data3=[];
for i=1:length(data1)
    if dataX(i)^2+dataY(i)^2 <= 1
        data3=cat(2,data3,data1(:,i));
    else
    end
end

dataX = data3(1,:).'; 
dataY = data3(2,:).';

[data_theta, dataR] = cart2pol(dataX, dataY);
polarscatter(data_theta,dataR);
%% 神经网络的初始配置过程

numLayers = 5;
numNeurons = 30;

layers = featureInputLayer(2);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(1)];
net = dlnetwork(layers)

solverState = lbfgsState;

Theta = dlarray(data_theta,"BC");              
R = dlarray(dataR,"BC");


theta0BC1 = dlarray(theta0BC1,"CB");
r0BC1 = dlarray(r0BC1,"CB");
BC1 = dlarray(BC1,"CB");

theta0BC2 = dlarray(theta0BC2,"CB");
r0BC2 = dlarray(r0BC2,"CB");
BC21 = dlarray(BC21,"CB");
BC2t = dlarray(BC2t,"CB");

lossFcn = @(net) dlfeval(@modelLoss,net,Theta,R,theta0BC1,r0BC1,BC1,theta0BC2,r0BC2,BC21,BC2t);

%% 神经网络训练过程
I=[];
loss=[];

for i = 1:8000
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    I(i)=i;
    loss(i)=solverState.Loss;

    fprintf('epoch：%d',i);
    fprintf('   loss：%d\n',loss(i));

    if i > 10
        loss_change = abs(loss(i) - loss(i-10));
        if loss_change == 0
            break;
        end
    end
    i=i+1;

end

save('h.mat',"net")

figure(1)
semilogy(I,loss,'r',LineWidth=1)
set(gca,'Fontname','Times New Roman','fontsize',12,'FontWeight','bold')
legend(["Error"])
