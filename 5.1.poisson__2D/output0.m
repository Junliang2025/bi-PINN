clc
clear all;
close all;

load('bi13.mat',"net")

numPredictions = 101;
xTest1 = linspace(0,1,numPredictions);
yTest1 = linspace(0,1,numPredictions);

Test1 = [];

for i = 1:numPredictions
    xTest = linspace(xTest1(i),xTest1(i),numPredictions);
    yTest = yTest1;

    Test = cat(1, xTest, yTest);
    Test1 = cat(2, Test1, Test);
end

Test1 = dlarray(Test1,'CB');

xTest2 = Test1(1,:);
yTest2 = Test1(2,:);

[u,ux,uy,Uxx,Uxy,Uyy,f,ureal,uxreal,uyreal]=  dlfeval(@rosenbrocks,xTest2,yTest2,net);

u = extractdata(u);
ureal = extractdata(ureal);
uxreal = extractdata(uxreal);
uyreal = extractdata(uyreal);

ux = extractdata(ux);
uy = extractdata(uy);
Uxx=extractdata(Uxx);
Uxy=extractdata(Uxy);
Uyy=extractdata(Uyy);

f=extractdata(f);


upred=reshape(u,numPredictions,numPredictions);
ureal=reshape(ureal,numPredictions,numPredictions);
uxpred=reshape(ux,numPredictions,numPredictions);
uxreal=reshape(uxreal,numPredictions,numPredictions);
uypred=reshape(uy,numPredictions,numPredictions);
uyreal=reshape(uyreal,numPredictions,numPredictions);
f_pred = reshape(f,numPredictions,numPredictions);

erroru=ureal-upred;
errorux=uxreal-uxpred;
erroruy=uyreal-uypred;

maxAbsError_U = max(abs([max(max(erroru)); min(min(erroru))]));
maxAbsError_Ux = max(abs([max(max(errorux)); min(min(errorux))]));
maxAbsError_Uy = max(abs([max(max(erroruy)); min(min(erroruy))]));

rmseu = sqrt(sum(erroru.^2,'all')/(numPredictions.^2));
rmseux = sqrt(sum(errorux.^2,'all')/(numPredictions.^2));
rmseuy = sqrt(sum(erroruy.^2,'all')/(numPredictions.^2));

fprintf('rmseu：%d\n',rmseu);
fprintf('rmseux：%d\n',rmseux);
fprintf('rmseuy：%d\n',rmseuy);

set(0,'defaultfigurecolor','w');

figure(1)
colormap("jet")
pcolor(xTest1,yTest1,upred)
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',12,'FontWeight','bold')
xlabel('\it{x}','FontWeight','bold')
ylabel('\it{y}','FontWeight','bold')

figure(2)
colormap("jet")
pcolor(xTest1,yTest1,ureal)
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',12,'FontWeight','bold')
xlabel('\it{x}','FontWeight','bold')
ylabel('\it{y}','FontWeight','bold')

figure(3)
colormap("jet")
pcolor(xTest1,yTest1,(abs(erroru)))
colorbar
shading interp
set(gca,'Fontname','Times New Roman','fontsize',12,'FontWeight','bold')
xlabel('\it{x}','FontWeight','bold')
ylabel('\it{y}','FontWeight','bold')




