clear;
close all;
load('crackforest.mat');
% No = 9;
sumbw = zeros(20,20,118);
subGT = zeros(400,118);
for No = 1:118
Im = crackIm{No};
GT = crackGT{No};
Tmax = graythresh(Im);
[counts,X] = imhist(Im);
Tmin = X(find(counts,1));
thresh = zeros(1,5);
flag = zeros(1,5);
subbw = zeros(20,20,5);
for th = Tmin:0.01:Tmax
    bw = Im < th;
    std_bw = std2(bw);
    if std_bw >= 0.04 && std_bw < 0.06 && flag(1) == 0
        thresh(1) = th;
        flag(1) = 1;
        subbw(:,:,1) = GCA(bw);
    elseif std_bw >= 0.06 && std_bw < 0.1 && flag(2) == 0
        thresh(2) = th;
        flag(2) = 1;
        subbw(:,:,2) = GCA(bw);
    elseif std_bw >= 0.1 && std_bw < 0.16 && flag(3) == 0
        thresh(3) = th;
        flag(3) = 1;
        subbw(:,:,3) = GCA(bw);
    elseif std_bw >= 0.16 && std_bw < 0.32 && flag(4) == 0
        thresh(4) = th;
        flag(4) = 1;
        subbw(:,:,4) = GCA(bw);
    elseif std_bw >= 0.32 && std_bw < 0.45 && flag(5) == 0
        thresh(5) = th;
        flag(5) = 1;
        subbw(:,:,5) = GCA(bw);
        break;
    end
end
sumbw(:,:,No) = sum(subbw,3);
subGT(:,No) = reshape(GCA(GT),400,1);
end
save('bw_sequence.mat','sumbw','subGT');
load('bw_sequence.mat');
opts.alpha = 1;
opts.batchsize = 5;
opts.numepochs = 100;

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

cnn = cnnsetup(cnn, sumbw(:,:,1:100), subGT(:,1:100));
cnn = cnntrain(cnn, sumbw(:,:,1:100), subGT(:,1:100), opts);

testbw = sumbw(:,:,103:104);
testGT = subGT(:,103:104);
net = cnnff(cnn,sumbw);
% figure;imshow(pixeldup(testbw(:,:,1)/5,8));
% figure;imshow(pixeldup(testbw(:,:,2)/5,8));
result_unfold = net.o;
result = zeros(320,480,118);
block_x = 320/20;
block_y = 480/20;
for i = 117:118
    Im = crackIm{i};
    GT = crackGT{i};
    figure;imshow(GT);
    result_temp = reshape(result_unfold(:,i),20,20);
    coff_block = zeros(320,480);
    for a = 1:20
        for b = 1:20
            coff_block((a-1)*block_x+1:a*block_x,(b-1)*block_y+1:b*block_y) = result_temp(a,b);
        end
    end
%     F1max = 0;
%     for a1 = -1:0.01:1
%         for a2 = -1:0.01:1
%             temp = (Im * a1) + (coff_block * a2) < 0;
%             TP = sum(sum(temp & GT));
%             FP = sum(sum(temp & ~GT));
%             FN = sum(sum(~temp & GT));
%             Pr = TP/(TP + FP);
%             Re = TP/(TP + FN);
%             F1 = 2 * Pr * Re/(Pr + Re);
%             if F1 > F1max
%                 F1max = F1;
%                 a1_best = a1;a2_best = a2;
%             end
%         end
%     end
    result(:,:,i) = (Im * 0.72) + (coff_block * -0.37) < 0;
    figure;imshow(result(:,:,i));
end

% a = reshape(result(:,1),20,20);
% b = reshape(result(:,2),20,20);
% figure;imshow(pixeldup(a,8));
% figure;imshow(pixeldup(b,8));
% GTa = reshape(testGT(:,1),20,20);
% GTb = reshape(testGT(:,2),20,20);
% figure;imshow(pixeldup(GTa,8));
% figure;imshow(pixeldup(GTb,8));

% [er, bad] = cnntest(cnn, sumbw, subGT);
% figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');