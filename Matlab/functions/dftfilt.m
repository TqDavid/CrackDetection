%该函数可接受图像与滤波函数，然后对图像进行滤波
function g=dftfilt(f,H)
F=fft2(f,size(H,1),size(H,2));
g=real(ifft2(H.*F));
g=g(1:size(f,1),1:size(f,2));
