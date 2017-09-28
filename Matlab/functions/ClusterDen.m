function base = ClusterDen(Im,thresh)
numlayer = length(thresh);                      %所用的层数
layer{1} = Im < thresh(1);                      %第1层点，也是基本层
base = layer{1};
% [L{1},numL] = bwlabel(base);                        %标记基本层的区域
% siz = numel(Im);
% numL = numL/siz;
% if numL < 6*10^(-4)
%     thresh = thresh(2:numlayer);
%     numlayer = 4;
%     layer{1} = Im < thresh(1);                      %第1层点，也是基本层
%     base = layer{1};
%     L{1} = bwlabel(base);                           %标记基本层的区域
% end
for i = 2:numlayer
    layer{i} = Im < thresh(i) & Im >= thresh(i - 1);    %计算每一层点，即处于两个阈值之间的点
%     L{i} = bwlabel(layer{i});                           %标记每一层的区域
    base = base + layer{i};                             %原有层加入新一层的点
    [L{i},num] = bwlabel(base);                         %对base标记
    D = regionprops(L{i},'Centroid');                   %计算标记区域的质心
    Centoridnew = cat(1, D.Centroid);                   %记录标记区域的质心
    ClusterDen = zeros(num,1);
    NumPos = 10 - i;
    for j = 1:num
        CluCen = Centoridnew(j,:);                      %读取标记区域的质心
        
        %首先提取满足要求的正方形
        rad = 2;                                        %聚类半径
        basepad = padarray(base, [rad rad], 0, 'both'); %填充需处理图像以便后续处理
        neibo = basepad(round(CluCen(2)):round(CluCen(2) + 2 * rad),round(CluCen(1)):round(CluCen(1) + 2 * rad));   %提取质心点的对应半径正方形邻域
        while(sum(neibo(:)) < NumPos)                       %如果正方形邻域内满足的点数小于设定值，扩大正方形的边长再次提取邻域
            rad = rad + 1;
            basepad = padarray(base, [rad rad], 0, 'both');
            neibo = basepad(round(CluCen(2)):round(CluCen(2) + 2 * rad),round(CluCen(1)):round(CluCen(1) + 2 * rad));
        end
        
        %计算相应圆内满足要求的点的个数，直至达到设定值
        [I,J] = find(neibo);
        distance = sqrt((I - (rad + 1)) .^ 2 + (J - (rad + 1)) .^ 2);%计算正方形邻域内点到中心点的距离
        count = sum(distance <= rad);                               %对中心距离小于聚类半径的点计数
        while(count < NumPos)
            rad = rad + 1;
            basepad = padarray(base, [rad rad], 0, 'both');
            neibo = basepad(round(CluCen(2)):round(CluCen(2) + 2 * rad),round(CluCen(1)):round(CluCen(1) + 2 * rad));
            [I,J] = find(neibo);
            distance = sqrt((I - (rad + 1)) .^ 2 + (J - (rad + 1)) .^ 2);
            count = sum(distance <= rad);
        end
        
        %判断是否为裂缝区域
        ClusterDen(j) = 1/rad;                  %聚类密度，定义为最小半径的倒数
        if ClusterDen(j) < 0.05 * 1.3 ^ i
            base(L{i} == j) = 0;
        end
    end
%     figure;hist(ClusterDen);
end
% TP = sum(sum(base & GT));
% FP = sum(sum(base & ~GT));
% FN = sum(sum(~base & GT));
% Pr = TP/(TP + FP)
% Re = TP/(TP + FN)
% F1 = 2 * Pr * Re/(Pr + Re)
% figure;imshow(base);