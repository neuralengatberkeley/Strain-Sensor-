function [RGB, sec] = yuv411Read(vid, width, height)

% yuv411read reads in raw images acquired in the yuv411 format -- it
% extracts the Y, U, V components and then converts to RGB at the end of
% the function

fid = fopen(vid,'r');           % Open the raw image
frame = fread(fid,'*uchar');    % Read the raw file
fN = size(frame, 1);
fclose(fid);

% Initialize u and v as zeros
u = gpuArray(uint8(zeros(height * width, 1)));
v = gpuArray(uint8(zeros(height * width, 1)));
N = height*width; 


%% Get the timestamps
timestamp = swapbytes(typecast(frame(1:4), 'uint32'));
timestamp = dec2bin(timestamp, 32);

secs = bin2dec(timestamp(1:7));
us125 = bin2dec(timestamp(8:20));
sec = double(secs) + double(us125)/8000;

%% Get the image: 
uix1 = (1:6:fN);
uix2 = (1:4:N);
upix = frame(uix1);

% using simple assignment bc of GPU array object
u(uix2) = upix;
u(uix2+1) = upix;
u(uix2+2) = upix;
u(uix2+3) = upix;

vix1 = (uix1 + 3);
vpix = frame(vix1);
v(uix2) = vpix;
v(uix2+1) = vpix;
v(uix2+2) = vpix;
v(uix2+3) = vpix;

uvix = [uix1 vix1];
yix = gpuArray(1:fN);
yix(uvix) = [];
y = gpuArray(frame(yix));

% Convert to RGB 
%input can be a GPU array
RGB = ycbcr2rgb(cat(3, reshape(y, [width, height])',...
    reshape(u, [width, height])',...
    reshape(v, [width, height])'));

% Get the data back from the GPU
RGB = gather(RGB);
clear stream frame
end