%imfreqfilt函数    对灰度图像进行频域滤波
%参数I             输入的时域图像
%参数ff            应用的与原图像等大的频域滤镜
%该函数参考《精通MATLAB数字图象处理与识别》第六章
function out=imfreqfilt(I,ff)
if((ndims(I)==3)&&size(I,3)==3)
    I=rbg2gray(I);
end
f=fft2(double(I));
s=fftshift(f);
out=s.*ff;
out=ifftshift(out);
out=ifft2(out);
out=abs(out);
out=out/max(out(:));
