# ADFCNN-MI
#Data Loader#

* BCI-IV2a：https://www.bbci.de/competition/iv/#datasets

* BCI-IV2b: https://www.bbci.de/competition/iv/#datasets

* OpenBMI: http://gigadb.org/dataset/view/id/100542/File_page

#Training#

* Set parameters from .config.ymal

* Set model from litmodel.py

* Run training.py

 #Testing#
 
* Set the correspongding config_name

* Load the model from check_path

* Run evaluation.py

#Suggested version of modules"

* python == 3.10.0

* torch == 1.12.0
# Welcome to cite this paper:
@article{tao2023adfcnn,
  title={ADFCNN: Attention-based dual-scale fusion convolutional neural network for motor imagery brain--computer interface},
  author={Tao, Wei and Wang, Ze and Wong, Chi Man and Jia, Ziyu and Li, Chang and Chen, Xun and Chen, CL Philip and Wan, Feng},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={32},
  pages={154--165},
  year={2024},
  publisher={IEEE}
}
