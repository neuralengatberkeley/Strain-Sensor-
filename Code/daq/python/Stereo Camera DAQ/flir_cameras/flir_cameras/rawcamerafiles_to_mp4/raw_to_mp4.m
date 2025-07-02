
%% Convert raw images to mp4
% select file example
% [files,paths] = uigetfile('*.raw', 'MultiSelect', 'on');

% Select folder from within SSD:
[filepath, name, ext] = fileparts(mfilename('fullpath'));
addpath([filepath filesep 'processing_utils']);
folder = uigetdir('C:/CAM_tmp/');


% cycle through all names in folder:        
files = dir(folder);
files_keep = {};
pref_unique = {};

% For real
save_dir = "S:";

% For testing! 
%save_dir = "C:\Users\scientist\Documents\test_camera";

for f = 1:length(files)
    
    % Get file name:
    nm = files(f).name;
    
    % remove the extension
    file_dashes = strfind(nm, '-');
    
    % if there are dashes, get the filename without the extension
    if ~isempty(file_dashes)
        
        nm_wo_ext = nm(1:file_dashes(end));
        add_pref = true;
        
        % Figure out if this raw file is from a new recording or a
        % recording that's already been opened
        for p = 1:length(pref_unique)
            if strcmp(pref_unique{p}, nm_wo_ext)
                add_pref = false;
            end
        end
        
        % If new recording, add the prefix to the list
        if add_pref
            pref_unique{end+1} = nm_wo_ext;
            files_keep{end+1} = [folder '\' nm];
        end
    end
end

files = files_keep;

if iscell(files)
    num_files = size(files, 2);
elseif ischar(files)
    num_files = 1;
    files = {files};
else
    exception('weird bug --files variable is not a cell and not a single filename!')
end

for i_f = 1:num_files
    file = files{i_f};
    
    disp(['Starting to process: ', file])
    
    if any(strfind(file, 'butters')) ||any(strfind(file, 'butters_silverTC')) || any(strfind(file, 'fifi'))|| any(strfind(file, 'granite'))% Behraz added:  Granite part
        % Get the dimensions of cam_side or cam_top
        if ~isempty(strfind(file, 'cam_side')) || ~isempty(strfind(file, 'fc2_save'))

            % Old coordinates used for camera -- somehow side cam got switched
            % back to these guys (PK, 8/27/19)
            row = 816;
            col = 692;

            % What was being used for side cam most recently
            %col = 856;
            %row = 896;

        elseif ~isempty(strfind(file, 'cam_top'))
            row = 880;
            %row = 1040;
            col = 550;
            %row = 768;
            %col = 800;
        else
            exception('didnt identify camera -- maybe file was saved with wrong name?')
        end
    else
        % Get the dimensions of cam_side or cam_top
        if ~isempty(strfind(file, 'cam_side'))

            % Old coordinates used for camera -- somehow side cam got switched
            % back to these guys (PK, 8/27/19)
            %row = 960;
            %col = 800;

            % What was being used for side cam most recently
            %col = 856;
            %row = 896;

%             col = 692; 
%             row = 816; 
            col = 766; 
            row = 1040; 
        elseif ~isempty(strfind(file, 'cam_top'))
%             row = 880;
            %row = 1040;
%             col = 550;
            row = 768;
            col = 800;
            
            %Spring 2022; useing butters' top cam params
            row = 880;
            %row = 1040;
            col = 550;
            
            row=960; 
            col=800; 
            
        else
            exception('didnt identify camera -- maybe file was saved with wrong name?')
        end
    end
    
    file_dashes = strfind(file, '-');
    dire = [file(1:file_dashes(end))];
    
    file_nonfold = strfind(file, '\');
    
    % Start the compressed video file for this recording
    vidname = strcat(save_dir, file(file_nonfold(end):file_dashes(end)-1), '.mp4');
    fname_final_ts = strcat(save_dir, file(file_nonfold(end):file_dashes(end)-1), '_timestamps.mat');
    
    cont = false; 
    if exist(fname_final_ts, 'file') ~= 2
        cont = true;
    end
    if exist(vidname, 'file') ~= 2
       cont = true;
    end
    
    if cont
        vid = VideoWriter(strcat(save_dir, file(file_nonfold(end):file_dashes(end)-1), '.mp4'), 'MPEG-4');
        vid.FrameRate = 50;
        open(vid);
    
        tic;
        timeStamps = [];
        tic1 = tic;
        frame = 0;
        continue_processing = true;

        % Loop through all the frames that are in the folder with this prefix:
        while continue_processing
            if frame == 0
                if exist([dire '000' num2str(frame) '.raw'], 'file')
                    ext = '.raw';
                elseif exist([dire '000' num2str(frame) '.jpg'], 'file')
                    ext = '.jpg';
                end
            end
            
            if frame < 10
                d = [dire '000' num2str(frame) ext];
            elseif frame < 100
                d = [dire '00' num2str(frame) ext];
            elseif frame < 1000
                d = [dire '0' num2str(frame) ext];
            else
                d = [dire num2str(frame) ext];
            end

            if exist(d, 'file')
                if strcmp(ext, '.raw')
                    % Convert the raw image to RGB + extract the embedded timestamp;
                    [rgb, sec] = yuv411Read(d, row, col);
                elseif strcmp(ext, '.jpg')
                    [rgb, ~] = imread(d);
                end

                % Write the frame to the video
                writeVideo(vid, gather(rgb));

                % Collect the timestamps
                if strcmp(ext, '.raw')
                    timeStamps = [timeStamps sec];
                end
                frame = frame + 1;
            else
                continue_processing = false;
            end

            if mod(frame, 100) == 0
                disp(['iter: ' num2str(frame)])
                disp('Time: ')
                toc(tic1)
                disp('')
                disp('')
                disp('')
                tic1 = tic;
            end
        end

        disp(['End of video: ', strcat(save_dir, file(1:file_dashes(end)-1), '.mp4')])
        disp(['Number of frames: ' num2str(frame)]);
        disp(['Approx secs: ', num2str(frame/50)]);
        
        if strcmp(ext, '.raw')
            overflows = [0 diff(timeStamps) < 0];
            times = timeStamps + cumsum(overflows)*128;
        end

        close(vid);
        if strcmp(ext, '.raw')
            %save(['S:' file(file_nonfold(end):file_dashes(end)-1) '_timestamps.mat'], 'times');
            fname_final_ts = strcat(save_dir, file(file_nonfold(end):file_dashes(end)-1), '_timestamps.mat');
            parsave_ts(fname_final_ts, times)
            %save([file(1:file_dashes(end)-1) '_timestamps.mat'], 'times');
        end
    else
        
        fprintf('Already saved video with this name: %s', vidname)
    end
end
