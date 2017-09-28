function base = DenFilter(Im,thresh)
rad = 5;
field = Circular(rad);
numlayer = length(thresh);                      %所用的层数
layer{1} = Im < thresh(1);                      %第1层点，也是基本层
base = layer{1};
[L{1},numL] = bwlabel(base);                           %标记基本层的区域
% siz = numel(Im);
% numL = numL/siz;
% if numL < 6*10^(-4)
%     thresh = thresh(2:numlayer);
%     numlayer = 4;
%     layer{1} = Im < thresh(1);                      %第1层点，也是基本层
%     base = layer{1};
%     [L{1},numL] = bwlabel(base);                           %标记基本层的区域
% end
basepad = padarray(base, [rad rad], 0, 'both');
D = regionprops(L{1},'Centroid');                %计算标记区域的质心
Centorid = cat(1, D.Centroid); 
for j = 1:numL
    CluCen = Centorid(j,:);
    neibo = basepad(round(CluCen(2)):round(CluCen(2) + 2 * rad),round(CluCen(1)):round(CluCen(1) + 2 * rad));
    den = sum(sum(neibo & field));
    if den < 4
        base(L{1} == j) = 0;
    end
end
% imwrite(base,['.\CrackForestPlot\Den\base',num2str(1),'.jpg'],'jpg');
for i = 2:numlayer
    layer{i} = Im < thresh(i) & Im >= thresh(i - 1);    %计算每一层点，即处于两个阈值之间的点
%     imwrite(layer{i},['.\CrackForestPlot\Den\layer',num2str(i),'.jpg'],'jpg');
    base = base + layer{i};                             %原有层加入新一层的点
%     imwrite(base,['.\CrackForestPlot\Den\base',num2str(i),'.jpg'],'jpg');
    basepad = padarray(base, [rad rad], 0, 'both');
    [L{i},numL] = bwlabel(base);                         %对base标记
    D = regionprops(L{i},'Centroid');                   %计算标记区域的质心
    Centorid = cat(1, D.Centroid); 
    for j = 1:numL
        CluCen = Centorid(j,:);
        neibo = basepad(round(CluCen(2)):round(CluCen(2) + 2 * rad),round(CluCen(1)):round(CluCen(1) + 2 * rad));
        den = sum(sum(neibo & field));
        if den < 5
            base(L{i} == j) = 0;
        end
    end
%     imwrite(base,['.\CrackForestPlot\Den\filtered',num2str(i),'.jpg'],'jpg');
%     figure;imshow(base);
end