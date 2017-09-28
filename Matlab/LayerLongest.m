%% 初始化
clear;
close all;
% load('AigleRN.mat');
load('crackforest.mat');
pics = 118;
Pr = zeros(1,pics);
Re = zeros(1,pics);
F1 = zeros(1,pics);
maxF1 = 0;
% maxF1 = 0.7253;
% R = 7;
% numBuild = 9;
% cof = 0.83;
% stdthresh = 0.19;
for stdthresh = 0.18:0.01:0.20
for numBuild = 8:10
for R = 6:8
for cof = 0.82:0.01:0.84
for No = 1:pics
% No = 3;
Im = crackIm{No};
GT = crackGT{No};
% Im = imread('2017.jpg');
% Im = im2double(rgb2gray(Im));
[M,N] = size(Im);

%% 预处理
Im = preproce(Im);
% figure;imshow(Im);

%% 获得合适的阈值
start = min(min(Im));
stop = graythresh(Im);
for thresh = start:0.01:stop
    bw = Im < thresh;
    entropy = std2(bw);
    if entropy > stdthresh
        L = bwlabel(bw);
        region = regionprops(L,'MajorAxisLength','MinorAxisLength','Orientation','BoundingBox');
        length = cat(1, region.MajorAxisLength);
        [length,index] = sort(length,'descend');
        break;
    end
end

%% 在得到合适阈值时，记录下当前阈值的最长区域
bw2 = zeros(size(Im));
if numel(index) > numBuild
    for i = 1:numBuild
        bw2(L == index(i)) = 1;
    end
else
    bw2 = bw;
end
% figure;imshow(bw2);

%% 裂缝搜索和连接
output = bw2;
[L,num] = bwlabel(output);
% R = 10;
for i = 1:num
    [x,y] = find(L == i,1);
    while x - R > 0  && y - R > 0 && x + R <= M && y + R <= N
        subblock = Im(x - R:x + R,y - R:y + R);
        meanblock = imfilter(subblock, 1/100 * ones(10,10));
        bwtemp = subblock < cof * meanblock;
        outtemp = output(x - R:x + R,y - R:y + R) | bwtemp;
        if sum(sum(outtemp - output(x - R:x + R,y - R:y + R))) == 0
            break;
        else
            output(x - R:x + R,y - R:y + R) = outtemp;
            [x1,y1] = find(outtemp,1);
            x = x1 + x - R;
            y = y1 + y - R;
        end
    end
    [x,y] = find(L == i,1,'last');
    while x - R > 0  && y - R > 0 && x + R <= M && y + R <= N
        subblock = Im(x - R:x + R,y - R:y + R);
        meanblock = imfilter(subblock, 1/100 * ones(10,10));
        bwtemp = subblock < cof * meanblock;
        outtemp = output(x - R:x + R,y - R:y + R) | bwtemp;
        if sum(sum(outtemp - output(x - R:x + R,y - R:y + R))) == 0
            break;
        else
            output(x - R:x + R,y - R:y + R) = outtemp;
            [x1,y1] = find(outtemp,1,'last');
            x = x1 + x - R;
            y = y1 + y - R;
        end
    end
end
output = imclose(output,strel('disk',1));
output = bwmorph(output,'spur',Inf);
output = bwareaopen(output,10);
% figure;imshow(output);
% figure;imshow(GT);
% [Pr, Re, F1] = score(output,GT);

[Pr(No), Re(No), F1(No)] = score(output,GT);
end
list = [Pr;Re;F1];
result = nanmean(F1)
if result > maxF1
    maxF1 = result;
    std_choose = stdthresh;
    build_choose = numBuild;
    R_choose = R;
    cof_choose = cof;
end
end
end
end
end