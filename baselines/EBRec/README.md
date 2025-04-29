# Enhancing Item-level Bundle Representation for Bundle Recommendation
## Requirements
* python == 3.7.3 or above
* supported(tested) CUDA versions: 10.2
* Pytorch == 1.9.0 or above
## How to run the code
```python
python train.py -g 0 -m EBRec -d NetEase -p "user_bundle_train"
python train.py -g 0 -m EBRec -d NetEase -p "Data-NetEase-(6,8)-[(5,5)]-2-1_Model-10-linear-0.1-0.1-1.0-20-128"
```
You can specify the gpu id and the used dataset by cmd line arguments, while you can tune the hyper-parameters by revising the configy file [config.yaml]. The detailed introduction of the hyper-parameters can be seen in the config file, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.
