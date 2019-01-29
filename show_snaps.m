clear all
close all

% Visualize wavefields
% List all files in the current directory
folder_name = './snap/';
dir_list = dir(folder_name);
% Tag which identifies a snapshot files (any unique key in the file name)
snap_tag = 'snap_';
% Get a logical vector that tells which files have snap_tag in their name
file_flags = strfind({dir_list.name},snap_tag);
nfiles = length(file_flags);
flags = zeros(1, nfiles);
for id = 1:nfiles
    if ~isempty(file_flags{id})
        flags(id) = 1;
    end
end
flags = logical(flags);
% Select only snap files
files = dir_list(flags);
% Sort folder names in the alphabetic order
% [~, reindex] = sort(str2double(regexp({files.name},'\d+', 'match', 'once' )));
nfiles = length(files);
fprintf('%i\t snaps\n', nfiles);
t_list = zeros(1, nfiles);
for ifile = 1:nfiles
    fname = files(ifile).name;
    dims = regexp(erase(fname,snap_tag),'\d*','Match');
    t_list(ifile) = str2double(dims{end-2});
    ny = str2double(dims{end-1});
    nx = str2double(dims{end});
end
[~, reindex] = sort(t_list);
t_list = t_list(reindex);       
files = files(reindex) ;
nfiles = length(files);
data_array = zeros(ny, nx, nfiles);
% For each snap file
for ifile = 1:nfiles
    fname = files(ifile).name;
    % get modeil dimensions from the snap name
    dims = regexp(erase(fname,snap_tag),'\d*','Match');
    nt = str2double(dims{end-2}); ny = str2double(dims{end-1}); nx = str2double(dims{end});
    fid = fopen([folder_name  fname],'r');
    bin = fread(fid,'float');
    data_array(:,:,ifile) = reshape(bin,[ny nx]);
    fclose(fid);
end

%%
h = figure;
for ifile = 1:nfiles
    data = squeeze(data_array(:,:,ifile));
    fname = files(ifile).name;
    fprintf('%s\t%i x %i\t%e %e\n',fname, ny, nx, min(data(:)), max(data(:)));
%     imagesc(data); colormap jet; axis equal tight; colorbar;
    title([num2str(t_list(ifile)) ' ' num2str(ny) ' x ' num2str(nx)]); drawnow;
%     surf(data); axis equal tight; colormap jet;
    surf(data); shading interp; lighting phong; colormap hot; 
    axis equal tight; colorbar; grid off; set(gcf, 'color', 'white');
    drawnow;
%     set(gcf,'Color', [0 0 0], 'Name', sprintf('Tiny FDTD, step = %i', n));
%     kk = waitforbuttonpress;


      % Capture the plot as an image 
%       frame = getframe(h); 
%       im = frame2im(frame); 
%       [imind,cm] = rgb2ind(im,256); 
%       % Write to the GIF File
%       if ifile>=10
%           if ifile == 10 
%               imwrite(imind,cm,'./doc/wave.gif','gif', 'Loopcount',inf,'DelayTime',0.1); 
%           else 
%               imwrite(imind,cm,'./doc/wave.gif','gif','WriteMode','append'); 
%           end 
%       end
end