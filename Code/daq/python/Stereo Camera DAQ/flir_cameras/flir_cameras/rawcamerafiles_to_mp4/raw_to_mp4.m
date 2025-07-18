% Select the parent folder that contains cam_side and cam_top
parent_dir = uigetdir('C:/flir_capture/', 'Select parent folder with camera folders');
cam_folders = {'cam_side', 'cam_top'};
frame_size = [1200, 1920];  % height x width from Python output
frame_rate = 50;

for i = 1:numel(cam_folders)
    cam_name = cam_folders{i};
    folder = fullfile(parent_dir, cam_name);
    raw_files = dir(fullfile(folder, [cam_name, '-*.raw']));
    raw_files = sort({raw_files.name});  % Simple filename sort

    if isempty(raw_files)
        warning('No .raw files found in %s', folder);
        continue;
    end

    fprintf('Processing %s\n', cam_name);
    output_video = fullfile(folder, [cam_name, '.mp4']);
    vid = VideoWriter(output_video, 'MPEG-4');
    vid.FrameRate = frame_rate;
    open(vid);

    for j = 1:numel(raw_files)
        fname = fullfile(folder, raw_files{j});
        fid = fopen(fname, 'rb');
        raw = fread(fid, prod(frame_size), '*uint8');
        fclose(fid);

        if numel(raw) ~= prod(frame_size)
            warning('File %s has wrong size, skipping', fname);
            continue;
        end

        % Reshape without transpose to preserve Bayer pattern
        gray_img = reshape(raw, fliplr(frame_size));  % [cols, rows] => 1920 x 1200

        % Demosaic Bayer pattern to RGB
        rgb = demosaic(gray_img, 'bggr');  % Try 'rggb' or 'grbg' if colors still look off

        writeVideo(vid, rgb);
    end

    close(vid);
    fprintf('Saved video to: %s\n', output_video);
end
