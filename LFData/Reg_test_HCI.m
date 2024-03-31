%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data from HCI dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uint8 0-255
% ['LFI_ycbcr']   [3,w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;

%% path
dataset = 'HCI';
folder = './HCI';
savepath = './Test/HCI/';
if exist(savepath, 'dir')==0
    mkdir(savepath);
end

listname = sprintf('./list/Test_HCI.txt');
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 512;
W = 512;

ah = 9;
aw = 9;

an_new = 7;
an_crop = ceil((ah - an_new) / 2);
%% initialization
% LFI_ycbcr = zeros(H, W, 3, ah, aw, 1, 'uint8');
% count = 0;

%% generate data
for k = 1:length(list)
    lfname = list{k};
    lf_path = fullfile(folder,lfname);
    disp(lf_path);
         
    img = zeros(ah,aw,H,W,3,'single');   
    for v = 1 : ah
        for u = 1 : aw
            ind = (v-1)*9+(u-1);
            imgname = strcat('input_Cam',num2str(ind,'%03d'),'.png');
            sub = im2double(imread(fullfile(lf_path,imgname)));
            %sub = rgb2ycbcr(sub);
            img(v,u,:,:,:) = sub(1:H,1:W,:);
        end
    end

    LF = img(an_crop+1:an_crop+an_new,an_crop+1:an_crop+an_new,:,:,:);
    save_path = [savepath, lfname, '.mat'];
    save(save_path, 'LF'); 
end  