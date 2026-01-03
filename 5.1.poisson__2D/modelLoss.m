function [loss,gradients] = modelLoss(net,X,Y,x0BC1,y0BC1,BC1,x0BC2,y0BC2,BC2,x0BC3,y0BC3,BC3,BC3y,x0BC4,y0BC4,BC4,BC4y)

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
f= Uxx + Uyy + (pi.^2-1).*(exp(X)).*sin(pi*Y);
zeroTargetf = zeros(size(f),"like",f);
mseF = l2loss(f,zeroTargetf);

%% mseU.
% x0BC1
XY1 = cat(1,x0BC1,y0BC1);
U0Pred1 = forward(net,XY1);

BCPred1g1=dlgradient(sum(U0Pred1,"all"),{x0BC1,y0BC1},EnableHigherDerivatives=true);
Ux1 = BCPred1g1{1};%Ux
%Uy1 = BCPred1g1{2};%Uy

mseU1 = l2loss(U0Pred1,BC1);

mseU1x = l2loss(Ux1,BC1);
msebU1 = mseU1x ;

%% x0BC2
XY2 = cat(1,x0BC2,y0BC2);
U0Pred2 = forward(net,XY2);

BCPred1g2=dlgradient(sum(U0Pred2,"all"),{x0BC2,y0BC2},EnableHigherDerivatives=true);
Ux2 = BCPred1g2{1};%Ux
%Uy2 = BCPred1g2{2};%Uy

mseU2 = l2loss(U0Pred2,BC2);

mseU2x = l2loss(Ux2,BC2);
msebU2 = mseU2x ;

%% x0BC3
XY3 = cat(1,x0BC3,y0BC3);
U0Pred3 = forward(net,XY3);

BCPred1g3=dlgradient(sum(U0Pred3,"all"),{x0BC3,y0BC3},EnableHigherDerivatives=true);
%Ux3 = BCPred1g3{1};%Ux
Uy3 = BCPred1g3{2};%Uy

mseU3 = l2loss(U0Pred3,BC3);
msebU3 = l2loss(Uy3,BC3y);

%% x0BC4
XY4 = cat(1,x0BC4,y0BC4);

BC4Pred = forward(net,XY4);
U0Pred4 = BC4Pred(1,:);

BCPred1g4=dlgradient(sum(U0Pred4,"all"),{x0BC4,y0BC4},EnableHigherDerivatives=true);
%Ux4 = BCPred1g4{1};%Ux
Uy4 = BCPred1g4{2};%Uy

mseU4 = l2loss(U0Pred4,BC4);
msebU4 = l2loss(Uy4,BC4y);

%%
mseU = mseU1 + mseU2 + mseU3 + mseU4;
mseB = msebU1 + msebU2 + msebU3 + msebU4;
loss = mseF + mseU + 1*mseB;

gradients = dlgradient(loss,net.Learnables);
end