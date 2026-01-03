function [loss,gradients] = modelLoss(net,X,Y,x0BC1,y0BC1,BC1,BC1y,x0BC2,y0BC2,BC2,BC2y,x0BC3,y0BC3,BC3,BC3n,x0BC4,y0BC4,BC4,x0BC5,y0BC5,BC5)

XY = cat(1,X,Y);               
U = forward(net,XY);

%% 自动微分
gradientsU = dlgradient(sum(U,"all"),{X,Y},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Uy = gradientsU{2};

gradientsUx = dlgradient(sum(Ux,"all"),{X,Y},EnableHigherDerivatives=true);
gradientsUy = dlgradient(sum(Uy,"all"),{X,Y},EnableHigherDerivatives=true);

Uxx = gradientsUx{1};
%Uxy = gradientsUx{2};
Uyy = gradientsUy{2};

%% mseF. 
f= 0.01.*(Uxx + Uyy);
zeroTargetf = zeros(size(f),"like",f);
mseF = l2loss(f,zeroTargetf);

%% x0BC1
XY1 = cat(1,x0BC1,y0BC1);

U0Pred1 = forward(net,XY1);
BCPred1h=dlgradient(sum(U0Pred1,"all"),{x0BC1,y0BC1},EnableHigherDerivatives=true);

Ux1 = BCPred1h{1};%Ux
Uy1 = BCPred1h{2};%Uy

mseU1 = l2loss(U0Pred1,BC1);
mseU1y = l2loss(Uy1,BC1y);
msebU1 = mseU1y ;

%% x0BC2
XY2 = cat(1,x0BC2,y0BC2);

U0Pred2 = forward(net,XY2);
BCPred1g=dlgradient(sum(U0Pred2,"all"),{x0BC2,y0BC2},EnableHigherDerivatives=true);

Ux2 = BCPred1g{1};%Ux
Uy2 = BCPred1g{2};%Uy

mseU2 = l2loss(U0Pred2,BC2);
mseU2y = l2loss(Uy2,BC2y);
msebU2 = mseU2y ;

%% x0BC3
XY3 = cat(1,x0BC3,y0BC3);

U0Pred3 = forward(net,XY3);
BCPred1gg=dlgradient(sum(U0Pred3,"all"),{x0BC3,y0BC3},EnableHigherDerivatives=true);
Ux3 = BCPred1gg{1};%Ux
Uy3 = BCPred1gg{2};%Uy

mseU3 = l2loss(U0Pred3,BC3);
b = 10*(-(y0BC3-0.5).*Ux3 + (x0BC3-0.5).*Uy3);
msebU3 = l2loss(b,BC3n);

%% xoBC4
XY4 = cat(1,x0BC4,y0BC4);

BC4Pred = forward(net,XY4);
U0Pred4 = BC4Pred(1,:);

BCPred1ggg=dlgradient(sum(U0Pred4,"all"),{x0BC4,y0BC4},EnableHigherDerivatives=true);
Ux4 = BCPred1ggg{1};%Ux
Uy4 = BCPred1ggg{2};%Uy

mseU4 = 0.01*l2loss(Uy4,BC4);
%% xoBC5
XY5 = cat(1,x0BC5,y0BC5);

BC5Pred = forward(net,XY5);
U0Pred5 = BC5Pred(1,:);

BCPred1g5=dlgradient(sum(U0Pred5,"all"),{x0BC5,y0BC5},EnableHigherDerivatives=true);

Ux5 = BCPred1g5{1};%Ux
Uy5 = BCPred1g5{2};%Uy

mseU5 = 0.01*l2loss(Uy5,BC5);

%%
mseU = mseU1 + mseU2 + mseU3 + mseU4 + mseU5;
mseB = msebU1 + msebU2 + msebU3;
loss = mseF + mseU + 1*mseB;

gradients = dlgradient(loss,net.Learnables);

end