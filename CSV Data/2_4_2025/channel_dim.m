% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.
img = imread('width images/IMG_0501.jpg');
imshow(img)

a = ginput(4);
pix_to_in = norm(a(1,2)-a(2,2)) / 0.125;

width = norm(a(4,:)-a(3,:)) / pix_to_in







 