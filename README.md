# 2016 OpenMIIR Representation Learning Experiment

This experiment using the public domain [OpenMIIR dataset](https://github.com/sstober/openmiir) has been described in the paper
[Sebastian Stober: *Learning Discriminative Features from Electroencephalography Recordings by Encoding Similarity Constraints.* In: Proceedings of 42nd IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP'17), 2017.](http://bib.sebastianstober.de/icassp2017.pdf)

Please cite this paper if you use any of this code!

Note that this is a revised and extended version of an experiment originally described in [Sebastian Stober; Avital Sternin; Adrian M. Owen & Jessica A. Grahn: *Deep Feature Learning for EEG Recordings.* In: arXiv preprint arXiv:1511.04306 2015.](http://arxiv.org/abs/1511.04306)


## Code Dependencies

This code heavily depends on [Theano](https://github.com/Theano/Theano), [Blocks](https://github.com/mila-udem/blocks) and [Fuel](https://github.com/mila-udem/fuel). Using CUDA/cuDNN is optional but strongly encouraged.
Further dependencies comprise [MNE-Python](https://github.com/mne-tools/mne-python/) for pre-processing and plotting, [librosa](https://github.com/librosa/librosa) for pre-processing, as well as "usual suspects" like numpy, scikit-learn, joblib etc.
Make sure these libraries are installed properly if you want to run this code!


## Project Structure

* data/  
Pre-processed data files. See separate README!

* deepthought/  
Deep-learning-related code. Refactored and extended version of the [legacy deepthought code](https://github.com/sstober/deepthought) that was based on the discontinued Pylearn2 framework.

* mneext/  
Alternative resample method for MNE-Python, copied from the [legacy deepthought code](https://github.com/sstober/deepthought).

* openmiir/  
Code specific to the OpenMIIR dataset, adapted from the [legacy deepthought code](https://github.com/sstober/deepthought).

* results/  
Pre-computed network parameters and results for the "Train..." jupyter notebooks. Directory names match with the job_id given in each notebook.


## Acknowledgments

This research was supported by the donation of a Geforce GTX Titan X graphics card from the NVIDIA Corporation.


## Contact

Sebastian Stober \<sstober AT uni-potsdam DOT de\>  
Research Focus Cognitive Sciences  
Machine Learning in Cognitive Science Lab  
University of Potsdam  
Potsdam, Germany  
  
[http://www.uni-potsdam.de/mlcog/](http://www.uni-potsdam.de/mlcog/)  