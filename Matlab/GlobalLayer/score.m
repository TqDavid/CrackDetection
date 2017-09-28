function [Pr, Re, F1] = score(output,GT)
TP = 0;
rad = 1;
% field = Circular(rad);
field = ones(3,3);
outpad = padarray(output, [rad rad], 0, 'both');
GTpad = padarray(GT, [rad rad], 0, 'both');
[x,y] = find(outpad);
for i = 1:length(x)
    neibo = GTpad(x(i) - rad:x(i) + rad,y(i) - rad:y(i) + rad);
    if sum(sum(neibo & field)) ~= 0
        TP = TP + 1;
    end
end
Pr = TP / length(x);
TP = 0;
[x,y] = find(GTpad);
for i = 1:length(x)
    neibo = outpad(x(i) - rad:x(i) + rad,y(i) - rad:y(i) + rad);
    if sum(sum(neibo & field)) ~= 0
        TP = TP + 1;
    end
end
Re = TP / length(x);
F1 = 2 * Pr * Re/(Pr + Re);