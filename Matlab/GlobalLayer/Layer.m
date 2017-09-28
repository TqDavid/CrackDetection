function thresh = Layer(Im)
iter = 6;                                       %迭代次数
numlayer = 5;                                   %使用层数
thresh = zeros(1,iter);                         
thresh(1) = graythresh(Im);                     %取全局阈值作为初始值
Imtemp = Im;
for redo = 1:iter                               %迭代二值处理
    bw = Imtemp > thresh(redo);                 %二值化
    Imtemp(bw) = thresh(redo);                  %用阈值取代大于阈值的像素
    thresh(redo + 1) = graythresh(Imtemp);      %再取全局阈值
end
thresh = fliplr(thresh);                        %水平翻转thresh数组
thresh = thresh(1:numlayer);                    %取前面几项作为研究，因为到原始全局阈值时已经噪声过度