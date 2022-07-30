## FMFP - Official PyTorch Implementation
![](./framework.png)

### [ECCV2022] Rethinking Confidence Calibration for Failure Prediction
Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu<br>

### Usage 
We run the code with torch version: 1.10.0, python version: 3.9.7
* Baseline and other calibration methods
```
python main_base.py
```
* Flat minima for failure prediction
```
python main_fmfp.py
```


### Reference
Our implementation references the codes in the following repositories:
* <https://github.com/daintlab/confidence-aware-learning>
* <https://github.com/davda54/sam>
* <https://github.com/timgaripov/swa>

### Contact
Fei Zhu (zhufei2018@ia.ac.cn)
