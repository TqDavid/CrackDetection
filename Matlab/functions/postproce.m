function output = postproce(base,Im)
output = base;

%% 消除圆率指数大的和面积小的连通域
output = FinalFilter(output);
output = bwareaopen(output,50);

%% 消除边界误判
output(:,1:5) = 0;
output(:,end-4:end) = 0;
output(1:5,:) = 0;
output(end-4:end,:) = 0;

%% 以当前裂缝为基准，区域阈值处理进行搜索
output = CrackSearch(output,Im);

%% 形态学处理，包括连接和消除小面积区域（搜索得到的错判区域）
output = bwmorph(output,'bridge',Inf);
output = imclose(output,strel('disk',2));
output = bwareaopen(output,10);