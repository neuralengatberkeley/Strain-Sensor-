function [rgb] = ycbcr2rgb_gpu(ycbcr)

% YUV is [height, width, 3]
Tinv = gpuArray([0.004566210045662, 1.180879989847092e-09, 0.006258928969944;
    0.004566210045662, -0.001536323686045, -0.003188110949656;
    0.004566210045662, 0.007910716233555, 1.197749704020026e-08]);

offset = gpuArray([16;128;128]);

scaleFactor.uint8.T = gpuArray([255]);        % scale output so it is in range [0 255].
scaleFactor.uint8.offset = gpuArray([255]);

T = scaleFactor.uint8.T * Tinv;
offset = scaleFactor.uint8.offset * Tinv * offset;
classIn = class(ycbcr);
rgb = zeros(size(ycbcr),classIn);

for p = 1:3
    rgb(:,:,p) = (T(p,1)*ycbcr(:,:,1)) + (T(p,2)*ycbcr(:,:,2)) + (T(p,3)*ycbcr(:,:,3)) - offset(p);
end

