clear;
close all;
load('crackforest.mat');
pics = 118;
Pr = zeros(1,pics);
Re = zeros(1,pics);
F1 = zeros(1,pics);
maxF1 = 0;
ThresholdDelta = 2.8;
% area = 5;
% for ThresholdDelta = 2.6:0.1:3.4
% for area = 1:5
for No = 1:pics
% No = 1;
Im = crackIm{No};
GT = crackGT{No};
Im = preproce(Im);
% figure;imshow(Im);
object = detectMSERFeatures(Im,'ThresholdDelta',ThresholdDelta);
region = cell(length(object),1);
bw = zeros(size(Im));
for i = 1:length(object)
    region{i} = object(i).PixelList;
    for j = 1:size(region{i},1)
        bw(region{i}(j,2),region{i}(j,1)) = 1;
    end
end
% figure;imshow(bw);
output = FinalFilter(bw);
% figure;imshow(output);
output = imclose(output,strel('disk',1));
% output = bwareaopen(output,area);
% output = imopen(output,strel('disk',1));
% figure;imshow(output);
output = bwmorph(output,'spur',Inf);

%% 去除边界错判
left = 0;
right = 0;
up = 0;
down = 0;
if sum(output(end,:)) > 0.25 * size(Im,2)
    down = 1;
elseif sum(output(1,:)) > 0.25 * size(Im,2)
    up = 1;
end
if sum(output(:,1)) > 0.25 * size(Im,1)
    left = 1;
elseif sum(output(:,end)) > 0.25 * size(Im,1)
    right = 1;
end
if up + down + left + down > 0
    line = 2;
    while up + down == 1 
        if down == 1 && sum(sum(output(end - line + 1:end,:))) > 0.2 * size(Im,2) * line && line < 5
            line = line + 1;
        elseif up == 1 && sum(sum(output(1:line,:))) > 0.2 * size(Im,2) * line && line < 5
            line = line + 1;
        else
            output(end - line + 1:end,:) = (1 - down) * output(end - line + 1:end,:);
            output(1:line,:) = (1 - up) * output(1:line,:);
            down = down + 1;
            up = up + 1;
        end
    end
    line = 2;
    while left + right == 1
        if left == 1 && sum(sum(output(:,1:line))) > 0.2 * size(Im,1) * line && line < 5
            line = line + 1;
        elseif right == 1 && sum(sum(output(:,end - line + 1:end))) > 0.2 * size(Im,1) * line && line < 5
            line = line + 1;
        else
            output(:,1:line) = (1 - left) * output(:,1:line);
            output(:,end - line + 1:end) = (1 - right) * output(:,end - line + 1:end);
            left = left + 1;
            right = right + 1;
        end
    end
end

%% 结果评估
% figure;imshow(output);
% figure;imshow(GT);
% [Pr, Re, F1] = score(output,GT);
[Pr(No), Re(No), F1(No)] = score(output,GT);
end
list = [Pr;Re;F1];
nanmean(F1)
% result = nanmean(F1);
% if result > maxF1
%     maxF1 = result;
%     area_choose = area;
%     th_choose = ThresholdDelta;
% end
% end
% end