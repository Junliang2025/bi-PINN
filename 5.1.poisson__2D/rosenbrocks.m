function [u,Ux,Uy,Uxx,Uxy,Uyy,f,ureal,uxreal,uyreal] = rosenbrocks(x,y,net)

D = forward(net,[x;y]);

u = D(1,:);

gradientsU = dlgradient(sum(u,"all"),{x,y},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Uy = gradientsU{2};

gradientsUx = dlgradient(sum(Ux,"all"),{x,y},EnableHigherDerivatives=true);
gradientsUy = dlgradient(sum(Uy,"all"),{x,y},EnableHigherDerivatives=true);

Uxx = gradientsUx{1};
Uxy = gradientsUx{2};
Uyy = gradientsUy{2};

f = Uxx + Uyy + (pi.^2-1).*(exp(x)).*sin(pi*y);

ureal = exp(x).*sin(pi.*y);
uxreal = exp(x).*sin(pi.*y);
uyreal = pi*exp(x).*cos(pi.*y);
end