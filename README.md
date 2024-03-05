# dugMatting-mindscore

**Training**:

1. Setup environment by `pip install -r requirement`.
2. Download the [P3M-10K](https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1) (Baidu Netdisk: [Link](https://pan.baidu.com/share/init?surl=X9OdopT41lK0pKWyj0qSEA)(pw: fgmc), [Agreement](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf) (MIT) ), and unzip to the directory of '*/data*'.
3. Modify the *config/ITMODNet_config.yaml* according to your experience (**Optional**). The *ITMODNet_config.yaml* contains the detailed comments for each parameter.
4. Run by `python train.py`

The *checkpoint* can be seen in '/checkSave/'.