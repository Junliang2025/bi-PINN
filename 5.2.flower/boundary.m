clc 
clear all
close all
theta = linspace(0,2*pi,100).';
 r1 = 1-0.2*cos(5*theta);


figure(1)
polarplot(theta,r1,'-','linewidth',1.5)

