function result = GCA(Im)
% block_size = 4;
% num_row = size(Im,1)/block_size;
% num_col = size(Im,2)/block_size;
num_row = 20;
num_col = 20;
block_x = size(Im,1)/num_row;
block_y = size(Im,2)/num_col;
result = zeros(num_row,num_col);
for a = 1:num_row
    for b = 1:num_col
        result(a,b) = sum(sum(Im((a-1)*block_x+1:a*block_x,(b-1)*block_y+1:b*block_y))) > 1/32 * block_x * block_y;     %每个元胞元素是4*4的像素矩阵
    end
end