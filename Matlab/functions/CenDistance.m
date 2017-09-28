function base = CenDistance(Im,thresh)
numlayer = length(thresh);
layer = cell(1,numlayer);
L = cell(1,numlayer);
layer{1} = Im < thresh(1);                      %第1层点，也是基本层
base = layer{1};
L{1} = bwlabel(base);                           %标记基本层的区域
D = regionprops(L{1},'Centroid','Boundingbox','MajorAxisLength');                %计算标记区域的质心
Centorid = cat(1, D.Centroid); 
Boundingbox = cat(1, D.BoundingBox);
for i = 2:numlayer
    layer{i} = Im < thresh(i) & Im >= thresh(i - 1);    %计算每一层点，即处于两个阈值之间的点
    L{i} = bwlabel(layer{i});                           %标记每一层的区域
    D = regionprops(L{i},'Centroid');                   %计算标记区域的质心
    Centoridnew = cat(1, D.Centroid);                   %新一层的质心
    layertemp = zeros(size(Im));
    for k = 1:size(Centoridnew,1)                        %如果这一层的点的质心与基本层相距较近，添加所有点到基本层
        for j = 1:size(Centorid,1)
            if norm(Centoridnew(k,:) - Centorid(j,:)) <= 2*max(Boundingbox(j,3),Boundingbox(j,4))
                layertemp(L{i} == k) = 1;
                break;
            end
        end
    end
    base = base + layertemp;                            %更新基本层
    L{1} = bwlabel(base);                           %标记基本层的区域
    D = regionprops(L{1},'Centroid','Boundingbox','MajorAxisLength');               %计算标记区域的质心
    Centorid = cat(1, D.Centroid);
    Boundingbox = cat(1, D.BoundingBox);
end