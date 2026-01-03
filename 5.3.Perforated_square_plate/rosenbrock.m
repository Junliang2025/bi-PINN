function [u,Ux,Uy,Uxx,Uxy,Uyy] = rosenbrock(x,y,net)



D = forward(net,[x;y]);


u = D(1,:);


% [Ux,Uy] = dlgradient(u,x,y);
% [Vx,Vy] = dlgradient(v,x,y);

gradientsU = dlgradient(sum(u,"all"),{x,y},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Uy = gradientsU{2};



gradientsUx = dlgradient(sum(Ux,"all"),{x,y},EnableHigherDerivatives=true);
gradientsUy = dlgradient(sum(Uy,"all"),{x,y},EnableHigherDerivatives=true);

Uxx = gradientsUx{1};
Uxy = gradientsUx{2};
Uyy = gradientsUy{2};




end