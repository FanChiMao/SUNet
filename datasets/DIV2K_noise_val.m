clc
clear 
close all

file_path = 'D:\NCHU\thesis\Ablation_study\Ablation_dataset\Set5\GT\';% 影象資料夾路徑 最後記得加\
target_path = 'D:\NCHU\thesis\Ablation_study\Ablation_dataset\Set5\noise_50\';
input_path = 'D:\PycharmProjects\SUNet-main\datasets\Denoising_DIV2K\test\input\';
img_path_list = dir(strcat(file_path,'*.png'));%獲取該資料夾中所有jpg格式的影象
img_num = length(img_path_list);%獲取影象總數量
count = 0;
if img_num > 0 %有滿足條件的影象   
   for j = 1:img_num %逐一讀取影象
       image_name = img_path_list(j).name;% 影象名
       input = imread(strcat(file_path,image_name));
       a = 255;
       b = 255;
       c = 3;%每?照片裁剪?量    
       X = size(input,1); 
       Y = size(input,2); 
       for c_img_num = 1:c
           y=randperm(X - (a + 1),1);
           x=randperm(Y - (b + 1),1);
           C = imcrop(input,[x y a b]);
           
           sig1 = 10; 
           V1 = (sig1/256)^2;
           sig2 = 30; 
           V2 = (sig2/256)^2;
           sig3 = 50; 
           V3 = (sig3/256)^2;

           added_noise1 = imnoise(C,'gaussian',0,V1);
           added_noise2 = imnoise(C,'gaussian',0,V2);
           added_noise3 = imnoise(C,'gaussian',0,V3);
           
           count = count + 1;
           imwrite(added_noise1,strcat(input_path, num2str(count),'.png'));
           imwrite(C,strcat(target_path, num2str(count),'.png'));
           
           count = count + 1;
           imwrite(added_noise2,strcat(input_path, num2str(count),'.png'));
           imwrite(C,strcat(target_path, num2str(count),'.png'));
           
           count = count + 1;
           imwrite(added_noise3,strcat(input_path, num2str(count),'.png'));
           imwrite(C,strcat(target_path, num2str(count),'.png'));

       end
       fprintf('Image %d\n', j) ;
    end
end
fprintf('finished!\n');