%% 初始化，读取图片
clear;
close all;
load('crackforest.mat');
%maxF1 = 0.7063;
pics = 118;
Pr = zeros(1,pics);
Re = zeros(1,pics);
F1 = zeros(1,pics);
tic;
for No = 1:pics
% No = 4;
Im = crackIm{No};
GT = crackGT{No};
% imwrite(Im,['.\Result\source',num2str(No),'.png'],'png');
% imwrite(GT,['.\Result\GT',num2str(No),'.png'],'png');
% figure;imshow(Im);

%% 预处理
Im = preproce(Im);
% figure;imshow(Im);

%% 确定用于分层的阈值
thresh = Layer(Im);
% thresh = Layer2(Im);

%% 从底层开始，逐层增加可疑区域
base = CenDistance(Im,thresh);                          %以每层连同域中心为中心，在中心周围生长可疑域
% base = DenFilter(Im,thresh);
% figure;imshow(base);

%% 图像后处理
output = postproce(base,Im);

%% 评估指标
% figure;imshow(output);
% figure;imshow(GT);
% [Pr, Re, F1] = score(output,GT);                            
% imwrite(output,['.\Result\output',num2str(No),'.png'],'png');
[Pr(No), Re(No), F1(No)] = score(output,GT);

end
list = [Pr;Re;F1];
nanmean(F1)
toc;