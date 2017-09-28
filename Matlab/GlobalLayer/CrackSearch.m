function output = CrackSearch(input,Im)
[M,N] = size(input);
output = input;
[L,num] = bwlabel(input);
R = 10;
count = 0;
for i = 1:num
    % 查找第一个裂缝点（左上）
    [x,y] = find(L == i,1);
    while x - R > 0  && y - R > 0 && x + R <= M && y + R <= N
        % 以裂缝点为中心划分出block
        subblock = Im(x - R:x + R,y - R:y + R);
        % 算出block的阈值信息
        meanblock = imfilter(subblock, 1/100 * ones(10,10));
        bwtemp = subblock < 0.66 * meanblock;
        % 通过阈值得到新的二值block
        outtemp = output(x - R:x + R,y - R:y + R) | bwtemp;
        if sum(sum(outtemp - output(x - R:x + R,y - R:y + R))) == 0
            break;
        else
            % 如果新的二值block有增加像素点，则把端点的位置设置为新的中心，重复以上操作
%             figure;
%             imshow(pixeldup(subblock,8));
%             figure;
%             imshow(output(x - R:x + R,y - R:y + R));
%             figure;
%             imshow(outtemp);
%             if count == 0
%                 imwrite(pixeldup(subblock,8), 'subblock.png', 'png');
%                 imwrite(pixeldup(output(x - R:x + R,y - R:y + R),8), 'outpre.png', 'png');
%                 imwrite(pixeldup(outtemp,8), 'outpost.png', 'png');
%             end
            output(x - R:x + R,y - R:y + R) = outtemp;
            [x1,y1] = find(outtemp,1);
            x = x1 + x - R;
            y = y1 + y - R;
%             if count == 0
%                 imwrite(pixeldup(output(x - R:x + R,y - R:y + R),8), 'outnext.png', 'png');
%                 count = count + 1;
%             end
%             figure;
%             imshow(output(x - R:x + R,y - R:y + R));
        end
    end
    [x,y] = find(L == i,1,'last');
    while x - R > 0  && y - R > 0 && x + R <= M && y + R <= N
        subblock = Im(x - R:x + R,y - R:y + R);
        meanblock = imfilter(subblock, 1/100 * ones(10,10));
        bwtemp = subblock < 0.66 * meanblock;
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