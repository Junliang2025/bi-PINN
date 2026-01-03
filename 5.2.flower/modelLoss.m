function [loss,gradients] = modelLoss(net,Theta,R,theta0BC1,r0BC1,BC1,theta0BC2,r0BC2,BC21,BC2t)
%%  
XY = cat(1,Theta,R);              
U = forward(net,XY);

%% 自动微分
gradientsU = dlgradient(sum(U,"all"),{Theta,R},EnableHigherDerivatives=true);
Ut = gradientsU{1};
Ur = gradientsU{2};

gradientsUt = dlgradient(sum(Ut,"all"),{Theta,R},EnableHigherDerivatives=true);
gradientsUr = dlgradient(sum(Ur,"all"),{Theta,R},EnableHigherDerivatives=true);

Utt = gradientsUt{1};
Urr = gradientsUr{2};
%Utr = gradientsUt{2};
%% mseF
f= Utt + Urr + 5.*cos(5.*Theta);
zeroTargetf = zeros(size(f),"like",f);
mseF = l2loss(f,zeroTargetf);

%% x0BC1
XY1 = cat(1,theta0BC1,r0BC1);
U0Pred1 = forward(net,XY1);
mseU1 = l2loss(U0Pred1,BC1);

%% x0BC2
XY2 = cat(1,theta0BC2,r0BC2);

U0Pred2 = forward(net,XY2);

BCPred1g2=dlgradient(sum(U0Pred2,"all"),{theta0BC2,r0BC2},EnableHigherDerivatives=true);
Ut2 = BCPred1g2{1};%Ux
Ur2 = BCPred1g2{2};%Uy

mseU2 = l2loss(U0Pred2,BC21);

mseU2t = l2loss(Ut2,BC2t);
msebU2 = mseU2t ;

%%
mseU = mseU1 + mseU2;
mseB = msebU2;
loss = mseF + mseU + 1.*mseB;
gradients = dlgradient(loss,net.Learnables);

end