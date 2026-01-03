clc
clear all
close all

%load('pinn.mat',"net")
%load('gpinn.mat',"net")
load('bipinn.mat',"net")

numPredictions = 101;
thetaTest1 = linspace(0,2*pi,numPredictions);
rTest1 = linspace(0,1,numPredictions);


Test1 = [];

for i = 1:numPredictions
    thetaTest = linspace(thetaTest1(i),thetaTest1(i),numPredictions);
    rTest = rTest1;

    Test = cat(1, thetaTest, rTest);
    Test1 = cat(2, Test1, Test);
end

Test1 = dlarray(Test1,'CB');

thetaTest2 = Test1(1,:);
rTest2 = Test1(2,:);

[u,utheta,ur,Utt,Utr,Urr,f,ureal]=  dlfeval(@rosenbrock,thetaTest2,rTest2,net);

upred = extractdata(u);
ureal = extractdata(ureal);
error=abs(ureal-upred).';
% uxreal = extractdata(uxreal);
% uyreal = extractdata(uyreal);

utheta = extractdata(utheta);
ur = extractdata(ur);
Utt=extractdata(Utt);
Utr=extractdata(Utr);
Urr=extractdata(Urr);

f=extractdata(f);

upred=reshape(upred,numPredictions,numPredictions);
ureal=reshape(ureal,numPredictions,numPredictions);
f_pred = reshape(f,numPredictions,numPredictions);

set(0,'defaultfigurecolor','w');

figure(1)
colormap("jet")
polarplot3d(upred, 'RadialRange', [0 1], 'AngularRange', [0 2*pi], 'PlotType', 'surfn');
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',14,'FontWeight','bold')
%view(2);

figure(2)
colormap("jet")
polarplot3d(ureal, 'RadialRange', [0 1], 'AngularRange', [0 2*pi], 'PlotType', 'surfn');
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',14,'FontWeight','bold')
%view(2);

figure(3)
colormap("jet")
polarplot3d(abs(ureal-upred), 'RadialRange', [0 1], 'AngularRange', [0 2*pi], 'PlotType', 'surfn');
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',14,'FontWeight','bold')
%view(2);
