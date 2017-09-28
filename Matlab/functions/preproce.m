function result = preproce(Im)
%% 去除不均匀背景
A=imresize(Im,[256 256]);
blocks=blkproc(A,[32,32],@estibackground);%分块处理
background=imresize(blocks,size(Im),'bilinear');%双线性插值进行缩放
A=imresize(A,size(Im));
A=imsubtract(A,background); %校正后

%% 增强对比度
Im = A;
Itop = imtophat(Im,strel('disk',15));           %顶帽变换
Ibottom = imbothat(Im,strel('disk',15));        %底帽变换
Im = Itop - Ibottom;                            %顶帽与底帽之差可增强对比度
result = Im + abs(min(min(Im)));                %增强对比度图像偏暗，把其灰度线性偏移