function field = Circular(rad)
edge = 2 * rad + 1;
field = ones(edge:edge);
for i = 1:edge
    for j = 1:edge
        if sqrt((i - rad - 1) ^ 2 + (j - rad - 1) ^ 2) > rad + 1
            field(i,j) = 0;
        end
    end
end