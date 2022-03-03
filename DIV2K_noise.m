clc
clear 
close all

file_path = 'D:\NCHU\Dataset\Denoise\DIV2K\DIV2K_train_HR\';% 影象資料夾路徑 最後記得加\
% D:\NCHU\Dataset\Denoise\DIV2K\DIV2K_valid_HR
img_path_list = dir(strcat(file_path,'*.png'));
img_num = length(img_path_list);%獲取影象總數量
count = 0;
if img_num > 0 %有滿足條件的影象   
   for j = 1:img_num %逐一讀取影象
       image_name = img_path_list(j).name;% 影象名
       input = imread(strcat(file_path,image_name));
       a = 255;
       b = 255;
       c = 100;
       X = size(input,1); 
       Y = size(input,2); 
       for c_img_num = 1:c
           y=randperm(X - (a + 1),1);
           x=randperm(Y - (b + 1),1);
           C = imcrop(input,[x y a b]);
           
           sig = round(rand(1,1)*45 + 5); 
           V = (sig/256)^2;
           added_noise = imnoise(C,'gaussian',0,V);
           
           count = count + 1;
           imwrite(C,strcat('D:\PycharmProjects\SUNet-main\datasets\Denoising_DIV2K\train\target\',num2str(count),'.png'));
           % D:\NCHU\1092\Image process\DIV2K\val\target(clean)
           imwrite(added_noise,strcat('D:\PycharmProjects\SUNet-main\datasets\Denoising_DIV2K\train\input\',num2str(count),'.png'));
           % D:\NCHU\1092\Image process\DIV2K\val\input(blur)
       end
       fprintf('Image %d\n', j) ;
    end
end
fprintf('finished!\n');