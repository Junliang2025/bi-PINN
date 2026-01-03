function [u,Utheta,Ur,Utt,Utr,Urr,f,ureal] = rosenbrock(theta,r,net)

D = forward(net,[theta;r]);

ureal = r - 1 + 0.2.*cos(5.*theta);
u = D(1,:);

gradientsU = dlgradient(sum(u,"all"),{theta,r},EnableHigherDerivatives=true);
Utheta = gradientsU{1};
Ur = gradientsU{2};

gradientsUtheta = dlgradient(sum(Utheta,"all"),{theta,r},EnableHigherDerivatives=true);
gradientsUr = dlgradient(sum(Ur,"all"),{theta,r},EnableHigherDerivatives=true);

Utt = gradientsUtheta{1};
Utr = gradientsUtheta{2};
Urr = gradientsUr{2};

f =  Utt + Urr + 5.*cos(5.*theta);

end