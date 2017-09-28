%该函数实现在频率矩形中计算任意点到指定点的距离
%fftshift函数将变换的原点移动到频率矩形的中心
function [U,V]=dftuv(M,N)
u=0:(M-1);
v=0:(N-1);
idx=find(u>M/2);
u(idx)=u(idx)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);