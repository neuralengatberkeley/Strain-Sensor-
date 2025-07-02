function [RGB, sec] = yuv411Read_rm2017(vid, width, height, newCbCr)

% yuv411read reads in raw images acquired in the yuv411 format -- it
% extracts the Y, U, V components and then converts to RGB at the end of
% the function

fid = fopen(vid,'r');           % Open the raw image
frame = fread(fid,'*uchar');    % Read the raw file
fN = size(frame, 1);
fclose(fid);

% Initialize u and v as zeros
u = uint8(zeros(height * width, 1));
v = uint8(zeros(height * width, 1));
N = height*width; 


%% Get the timestamps
timestamp = swapbytes(typecast(frame(1:4), 'uint32'));
timestamp = dec2bin(timestamp, 32);

secs = bin2dec(timestamp(1:7));
us125 = bin2dec(timestamp(8:20));
sec = double(secs) + double(us125)/8000;

%% Get the image: 
if newCbCr
    ix4 = 1:4:N; 
    
    uix = 3:6:fN; 
    upix = frame(uix); 
    u(ix4) = upix;
    u(ix4+1) = upix;
    u(ix4+2) = upix; 
    u(ix4+3) = upix; 
    
    vix = 6:6:fN; 
    vpix = frame(vix); 
    v(ix4) = vpix;
    v(ix4+1) = vpix;
    v(ix4+2) = vpix; 
    v(ix4+3) = vpix; 
    
    yix = 1:fN; 
    uv = [uix vix]; 
    yix(uv) = []; 
    y = frame(yix); 
    
else
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
    yix = (1:fN);
    yix(uvix) = [];
    y = (frame(yix));
end
% Convert to RGB 
%input can be a GPU array

% For YCbCR --> var u = y; 
% No idea what U/V are


RGB = ycbcr2rgb(cat(3, reshape(y, [width, height])',...
    reshape(u, [width, height])',...
    reshape(v, [width, height])'));

% Get the data back from the GPU
clear stream frame
end