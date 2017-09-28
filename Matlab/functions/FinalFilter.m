function output = FinalFilter(base)
[L{1}, numL] = bwlabel(base);
D = regionprops(L{1},'ConvexArea','Boundingbox','MajorAxisLength','Perimeter');
Perimeter = cat(1, D.Perimeter);
Area = cat(1, D.ConvexArea);                  %标记区域面积
MajorLength = cat(1, D.MajorAxisLength);
den = (4 * Area) ./ (pi * MajorLength .^ 2);
den2 = Area ./ (Perimeter .^ 2);
for i = 1:numL
    if den(i) > 0.45 || den2(i) > 0.05
        base(L{1} == i) = 0;
    end
%     if den2(i) > 0.025
%         base(L{1} == i) = 0;
%     end
end
output = base;