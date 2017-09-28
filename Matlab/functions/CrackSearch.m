function output = CrackSearch(input,Im)
[M,N] = size(input);
output = input;
[L,num] = bwlabel(input);
R = 10;
for i = 1:num
    [x,y] = find(L == i,1);
    while x - R > 0  && y - R > 0 && x + R <= M && y + R <= N
        subblock = Im(x - R:x + R,y - R:y + R);
        meanblock = imfilter(subblock, 1/100 * ones(10,10));
        bwtemp = subblock < 0.66 * meanblock;
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