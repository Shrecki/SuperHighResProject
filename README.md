# SuperHighResProject
Repository of "Learning for super resolution" project 2018

The purpose of this project was to compare two approaches to deep learning based
super resolution, one based on wavelet transforms and the other on the spatial domain.
Models were compared with respect to each other as well as to bicubic interpolation through
measures such as PSNR, SSIM and RMSE. It was shown that networks trained on the frequency
and spatial domain outperformed bicubic interpolation and the two had very similar performance with wavelets achieving a slightly higher performance. 

We trainned two networks in wavelet and spatial domain using residual netkorks and keras.

All image processing methods are in the file "srPreprocessing.py"
Networks architecture is implemented in srcnn.py and wavelet_cnn.py in spatial and wavelets domain respectively
### Two notebooks for each model pipeline from the high definition image, trainning the models to predicting results 
  #### Preprocessing and Netbook training in wavelet domain pipeline :
    SRCNN_notebook.ipynb
  #### Preprocessing and Netbook training in spatial domain pipeline :
	  SRCNN_spatial_notebook.ipynb
### Notebook with all metrics used to compare, examples, and results:
  Comparison.ipynb
  
Motivations, discussion and results are
  "report" folder
  
### References 
Learning a Deep Convolutional Network for Image Super-Resolution, Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang 

Accurate Image Super-Resolution Using Very Deep Convolutional Networks Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee

Deep Wavelet Prediction for Image Super-resolution Tiantong Guo, Hojjat Seyed Mousavi, Tiep Huu Vu, Vishal Monga

J. Simpkins, R.L. Stevenson, "An Introduction to Super-Resolution Imaging." Mathematical Optics: Classical, Quantum, and Computational Methods, Ed. V. Lakshminarayanan, M. Calvo, and T. Alieva. CRC Press, 2012. 539-564.

