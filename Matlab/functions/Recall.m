function output = Recall(bw,Im)
sharpen = fspecial('log',3,0.5);
output = imfilter(bw,sharpen,'corr','replicate');
se = strel('square',3);
Ie = imerode(output,se);
bound = output - Ie;
[x,y] = find(output);
maxwin = 3;
for i = 1:length(x)
    win = 2;
    while win <= maxwin
        Impad = padarray(Im, [win win], 'replicate', 'both');
        outpad = padarray(output, [win win], 0, 'both');
        neibo = Impad(x(i):x(i) + 2 * win, y(i):y(i) + 2 * win);
        thresh = graythresh(neibo);
        neibo = neibo < thresh;
        outneibo = outpad(x(i):x(i) + 2 * win, y(i):y(i) + 2 * win);
        if sum(sum(~neibo & outneibo)) == 0
            break;
        end
        outpad(x(i):x(i) + 2 * win, y(i):y(i) + 2 * win) = outneibo | neibo;
        output = outpad(win + 1:end - win,win + 1: end - win);
        win = win + 1;
    end
end