function output = Recall2(bw,Im)
[L, num] = bwlabel(bw);
D = regionprops(L,'Boundingbox');
Boundingbox = cat(1, D.BoundingBox);
Boundingbox(:,1:2) = Boundingbox(:,1:2) + 0.5;
for i = 1:num -1
    j = i + 1;
    upx = min(Boundingbox(i,2),Boundingbox(j,2));   %上边x坐标
    lefty = min(Boundingbox(i,1),Boundingbox(j,1)); %左边y坐标
    if upx == Boundingbox(i,2)                      %根据情况，得到其余坐标
        upy = Boundingbox(i,1);
        downx = Boundingbox(j,2);
        downy = Boundingbox(j,1);
    else
        upy = Boundingbox(j,1);
        downx = Boundingbox(i,2);
        downy = Boundingbox(i,1);
    end
    if norm(upx - downx,upy - downy) < 10
        if upy == lefty                                 %左上角和右下角的情况
            subim = Im(upx:downx,upy:downy);
        else                                            %左下角和右上角的情况
            subim = Im(upx:downx,downy:upy);
        end
%         thresh = imfilter(subim,1/numel(subim)*ones(size(subim)));
        thresh = imfilter(subim,1/400*ones(20,20));
        subw = subim < thresh;
        if upy == lefty                                 %左上角和右下角的情况
            bw(upx:downx,upy:downy) = bw(upx:downx,upy:downy) | subw;
        else                                            %左下角和右上角的情况
            bw(upx:downx,downy:upy) = bw(upx:downx,downy:upy) | subw;
        end
%         bw = FinalFilter(bw);
    end
end
bw = FinalFilter(bw);
output = bw;