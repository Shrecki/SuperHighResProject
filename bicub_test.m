scale=3;
baboon_hr = imread('set14/img_001_SRF_4_HR.png');
baboon_lr= imresize(baboon_hr,1/scale);
baboon_bic=imresize(baboon_lr,[480 500]);

disp(psnr(baboon_bic,baboon_hr))


baboon_hr_ycbcr = rgb2ycbcr(baboon_hr);
baboon_hr_ycbcr=baboon_hr_ycbcr(:,:,1);
baboon_lr_ycbcr = imresize(baboon_hr_ycbcr,scale);
baboon_bic_ycbcr=imresize(baboon_lr_ycbcr,[480,500]);

disp(psnr(baboon_bic_ycbcr,baboon_hr_ycbcr))

