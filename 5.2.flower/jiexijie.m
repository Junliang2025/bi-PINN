clc;
clear all;

theta = linspace(0, 2*pi, 1001); 
r = linspace(0, 1, 1001);
[Theta, R] = meshgrid(theta, r);
Z = (R-(1-0.2.*cos(5.*Theta))); 


figure(1);
colormap("jet")
polarplot3d(Z, 'RadialRange', [0 1], 'AngularRange', [0 2*pi], 'PlotType', 'surfn');
shading interp; 
colorbar;
%view(2); 

