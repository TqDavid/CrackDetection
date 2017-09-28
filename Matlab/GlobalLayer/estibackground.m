function backgray=estibackground(x)
thr = 1;
meanx=mean(x(:));
stdx=std(x(:));
minx=min(x(:));
backgray=max(meanx-thr*stdx,minx);
end
