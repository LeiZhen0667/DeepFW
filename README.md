# DeepFW: A DNN-Based Framework for Firmware Version Identification of Online IoT Devices

With the rapid ubiquity of Internet of Things (IoT) technology, a growing number of devices are being connected to the Internet, thereby increasing the potential for cyberattacks. For instance, due to firmware compatibility issues and release delays, $N$-day vulnerabilities pose significant threats to IoT devices that run outdated firmware versions. Consequently, accurately and efficiently identifying firmware versions of devices is crucial for detecting device vulnerabilities and enhancing the security of IoT ecosystems. 
In this work, we present DeepFW, which utilizes a Fusion Feature Attention Network (FFAN) to extract subtle differences in embedded web interfaces within the firmware, facilitating the identification of firmware versions in online IoT devices. To address the challenge of high similarity between versions caused by firmware homogeneity in the supply chain, we propose a novel metric loss, namely the Hard Mining Cosine Triplet-Center Loss (HCTCL), to improve intra-class compactness and inter-class separability. To validate the effectiveness of our method, we collected 4,442 firmware images and obtained 130,445 valid embedded web pages. Experimental results show that DeepFW outperforms the state-of-the-art approaches by over 25\% on average in both precision and recall. Furthermore, DeepFW revealed that only 2.28% of devices in our dataset were running the latest firmware version. Our evaluation also indicates that 6,684 devices (approximately 61.26%) with outdated firmware versions remain vulnerable to known exploits.

Due to the sensitivity of data sourced from firmware or online devices, we provide only simplified pretrained models and code in the repository to facilitate the reproduction of our work.

## Installation

1. Clone the repository：
    ```bash
    git clone git@github.com:LeiZhen0667/DeepFW.git
    cd DeepFW
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a version of PyTorch with CUDA support installed if using a GPU.

## Usage Instructions

### Data Preparation

1. The dataset is stored in `dataset_demo`, containing embedded web pages from online IoT devices, from six firmware versions of four manufacturers: D-Link, TP-LINK, Linksys, and Netgear. Both this demonstration dataset and the pre-trained model are downloadable via the [link](https://drive.google.com/file/d/1ymeBUA89LQVOEiBeLQHEDVaRspiDBBiM/view?usp=sharing).
2. The `data_utils.py` file contains some utility functions and classes for data preprocessing and loading. These functionalities are mainly used for processing and preparing text data, especially for using pre-trained word vector models (such as GloVe) and custom datasets.

### Model Definition

1. Model definitions are located in the `models` directory. Key files include:
-  `FFAN.py`: Includes multi-scale convolutional layers, self-attention layers, and custom state updating layers for feature extraction.
-  `classifier.py`: Fully connected layers + softmax for final classification.
2. Loss functions are defined in the `losses` directory, with a key file:
- `HCTCL.py`: Custom Hard Mining Cosine Triplet-Center Loss for optimizing model training.

### Testing the Model

Use `test.py` to test the trained model. This module loads the saved model weights and evaluates model performance.
```bash
python test.py
```
The test results include the classification accuracy of the model.

### Training the Model

In `train.py`, configure model parameters and training settings such as learning rate, batch size, and number of training epochs. Run the script to perform model training and comparison with state-of-the-art approaches 
- when comparing, select 'both' for the `--method` option. Additionally, adjust the `--pool_sizes` parameter to configure version scale selection.

```bash
python train.py --root_dir dataset_demo --glove_path Static_word_embedding_pre-trained_models/glove.6B.300d.txt --method deepfw  --pool_sizes [8,16,32] --output_dir experiment_results
```


## Notes

- Ensure consistent `device` settings across all scripts to avoid tensor mismatch issues between GPU/CPU.
- Data file paths should be correctly set in the code.
- When using a GPU for training and testing, ensure CUDA is available and properly configured.





