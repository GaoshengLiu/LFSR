%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data from SIGGRAPHAsia16_ViewSynthesis_Kalantari_Trainingset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output: train_SIG.h5  
% uint8 0-255
% ['LFI']   [w,h,aw,ah,N]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
savepath = './Test/30scenes/';
if exist(savepath, 'dir')==0
    mkdir(savepath);
end
folder = './SIG/30scenes';

listname = './list/Test_30scenes.txt';
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 374;
W = 540;  

allah = 14;
allaw = 14;

ah = 7;
aw = 7;
an_crop = ceil((allah - ah) / 2 );
%% initialization
%LF = zeros(ah, aw, H, W,  3, 'single');

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = sprintf('%s/%s.png',folder,lfname);
    disp(lf_path);
    
    %eslf = im2single(imread(lf_path));
    eslf = single(im2uint8(imread(lf_path)))/255;
    img = zeros(allah,allaw,H,W,3,'single');    

    for v = 1 : allah
        for u = 1 : allah            
            sub = eslf(v:allah:end,u:allah:end,:);            
            %sub = rgb2ycbcr(sub);           
            img(v,u,:,:,:) = sub(1:H,1:W,:);        
        end
    end
        
    LF = img(an_crop:ah+an_crop,an_crop:aw+an_crop,:,:,:);

    %LF(:, :, :, :, count) = img;
    save_path = [savepath, lfname, '.mat'];
    save(save_path, 'LF'); 
end  
 

