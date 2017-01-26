Pre-processed data files are placed here.

Run “1 RAW Dataset Preparation 512Hz.ipynb“ to generate OpenMIIR-Perception-512Hz.pklz containing variable-length trials and their meta-data. This is an intermediate file needed for the next step but not for training.

Run “2 Make HDF5 Dataset.ipynb” to generate OpenMIIR-Perception-512Hz.hdf5 containing same-length trials (length of shortest one) ready to be used with Blocks and OpenMIIR-Perception-512Hz.hdf5.meta.pklz containing their meta-data.

Alternatively, these two files can be downloaded from http://www.ling.uni-potsdam.de/mlcog/OpenMIIR/rl2016/data/