clc
clear
close all
format long
rand('seed',3);

%% 边界条件
num = 13;
numBoundary = [num num num num];

x0BC1 = linspace(0,1,numBoundary(1));
y0BC1 = linspace(0,0,numBoundary(1));
BC1 = zeros(1,numBoundary(1));%u(x,0)=0

x0BC2 = linspace(0,1,numBoundary(2));
y0BC2 = linspace(1,1,numBoundary(2));
BC2 = zeros(1,numBoundary(2));%u(x,1)=0

x0BC3 = linspace(0,0,numBoundary(3));
y0BC3 = linspace(0,1,numBoundary(3));
BC3 = sin(pi*y0BC3);%u(0,y)
BC3y = pi*cos(pi*y0BC3);%uy(0,y)

x0BC4 = linspace(1,1,numBoundary(4));
y0BC4 = linspace(0,1,numBoundary(4));
BC4 = exp(1)*sin(pi*y0BC4);%u(1,y)
BC4y = pi*exp(1)*cos(pi*y0BC4);%uy(1,y)

%% 内部 
x1 = linspace(0,1,num);
data1 = [];
for i=1:num
    x2 = repmat(x1(i),1,num);
    y2 = linspace(0,1,num);
    data2 = [x2;y2];
    data1 = cat(2,data1,data2);
end

dataX = data1(1,:).'; 
dataY = data1(2,:).';

scatter(dataX,dataY)
%% 
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

%%                   
solverState = lbfgsState;

X = dlarray(dataX,"BC");               
Y = dlarray(dataY,"BC");

x0BC1 = dlarray(x0BC1,"CB");
y0BC1 = dlarray(y0BC1,"CB");
BC1 = dlarray(BC1,"CB");

x0BC2 = dlarray(x0BC2,"CB");
y0BC2 = dlarray(y0BC2,"CB");
BC2 = dlarray(BC2,"CB");

x0BC3 = dlarray(x0BC3,"CB");
y0BC3 = dlarray(y0BC3,"CB");
BC3 = dlarray(BC3,"CB");
BC3y = dlarray(BC3y,"CB");

x0BC4 = dlarray(x0BC4,"CB");
y0BC4 = dlarray(y0BC4,"CB");
BC4 = dlarray(BC4,"CB");
BC4y = dlarray(BC4y,"CB");

lossFcn = @(net) dlfeval(@modelLoss,net,X,Y,x0BC1,y0BC1,BC1,x0BC2,y0BC2,BC2,x0BC3,y0BC3,BC3,BC3y,x0BC4,y0BC4,BC4,BC4y);

%% 
I=[];
loss=[];

for i = 1:5000
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

save('bi13.mat',"net")

figure(1)
semilogy(I,loss,'r',LineWidth=1)
set(gca,'Fontname','Times New Roman','fontsize',12,'FontWeight','bold')
legend(["Error"])

