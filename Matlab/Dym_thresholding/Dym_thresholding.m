%% 初始化，读取图片
clear;
close all;
load('crackforest.mat');                        %加载CrackForest数据库
%参数
% maxF1 = 7103;
cof = 0.72;
block_size = 9;
numL = 8;
pics = 118;
Pr = zeros(1,pics);
Re = zeros(1,pics);
F1 = zeros(1,pics);
tic;
for No = 1:pics
% No = 1;
Im = crackIm{No};                               %读取图片
GT = crackGT{No};                               %读取GroundTruth
% Im = imread('2017.jpg');
% Im = im2double(rgb2gray(Im));
[M,N] = size(Im);
% figure;imshow(Im);
% imwrite(Im,['.\Result\source',num2str(No),'.png'],'png');

%% 预处理
Im = preproce(Im);                              %消除不均匀背景和增强对比度
% figure;imshow(Im);
% imwrite(Im,['.\Result\source',num2str(No),'.png'],'png');

%% 动态阈值处理
template = fspecial('average',2 * block_size + 1);
meanIm = imfilter(Im, template); %以19*19窗口邻域平均值作为局部信息
% figure;imshow(cof * meanIm);
% imwrite(meanIm,['.\Result\mean',num2str(No),'.png'],'png');
output = Im < cof * meanIm;                     %当灰度小于局部阈值时，认为是裂缝可疑点
% figure;imshow(output);
% imwrite(output,['.\Result\filtered',num2str(No),'.png'],'png');

%% 后处理
output = bwmorph(output,'bridge',Inf);
output = imclose(output,strel('disk',2));       %对所得区域进行形态学桥接

[L,num] = bwlabel(output);                      %保留输出中最长的区域
region = regionprops(L,'MajorAxisLength');
leng = cat(1, region.MajorAxisLength);
[leng,index] = sort(leng,'descend');
output = zeros(size(Im));
if length(leng) > numL
    j = 1;
    while j < numL
        if sum(leng(1:j)) > 1.2 * max([M,N])
            for i = 1:j
                output(L == index(i)) = 1;
            end
            break;
        else
            j = j + 1;
        end
    end
    if j == numL
        for i = 1:j
            output(L == index(i)) = 1;
        end
    end
else
    output = base;
end

output = FinalFilter(output);                   %去除圆率指数大的区域（去干扰）
% imwrite(output,['.\Result\output',num2str(No),'.png'],'png');

%% 评估指标
% figure;imshow(output);
% figure;imshow(GT);
% [Pr, Re, F1] = score(output,GT);

[Pr(No), Re(No), F1(No)] = score(output,GT);    %以区域为标准计算Pr,Re和F1
end
list = [Pr;Re;F1];
result = nanmean(F1)
time = toc / 118
% if result > maxF1
%     maxF1 = result;
%     cof_choose = cof;
%     block_choose = block_size;
%     numL_choose = numL;
% end
% end
% end
% end
