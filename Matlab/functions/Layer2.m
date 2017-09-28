function thresh = Layer2(Im)
start = min(min(Im)) + 0.01;
stop = graythresh(Im);
thresh = zeros(1,4);
flag = zeros(1,4);
for th = start:0.03:stop
    bw = Im < th;
    std_bw = std2(bw);
    if std_bw >= 0.05 && std_bw < 0.1 && flag(1) == 0
        thresh(1) = th;
        flag(1) = 1;
    elseif std_bw >= 0.1 && std_bw < 0.16 && flag(2) == 0
        thresh(2) = th;
        flag(2) = 1;
    elseif std_bw >= 0.16 && std_bw < 0.32 && flag(3) == 0
        thresh(3) = th;
        flag(3) = 1;
    elseif std_bw >= 0.32 && flag(4) == 0
        thresh(4) = th;
        flag(4) = 1;
        break;
    end
end