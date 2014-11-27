function [  ] = plot_data( X, width )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% This program will display images in a big image
% X is the picture arranged in a row. 

[n,m] = size(X);
if ~exist('width','var') || isempty(width)
    width = ceil(sqrt(m));
end;

%% parameter for displaying 
%height of each image
height = m./width;
%number of images in a column
column_num = ceil(sqrt(n));
row_num = ceil(n./column_num);
% fill the empty place with 0
empty = row_num * column_num - n;
X = [X;zeros(empty,m)];
%gap between images
pad = 1;

%% copy each image into the large image
colormap(gray);
im = -ones((pad + (height + pad)*row_num),pad + (width + pad)*column_num);
it = 1;
for row = 1:row_num
    for column = 1:column_num
        a = reshape(X(it,:),height,width);
        it = it + 1;
        im((1+(pad + height) *(row -1) + pad):((pad + height)*row),(1+(pad+width)*(column -1)+pad):((pad + width)*column)) = a;
    end;
end;
imagesc(im);


end

