# DeepFW: A DNN-Based Framework for Firmware Identification of Online IoT Devices

With the rapid ubiquity of Internet of Things (IoT) technology, a growing number of devices are being connected to the internet, thereby increasing the potential for cyberattacks. For instance, due to firmware compatibility issues and release delays, $N$-day vulnerabilities pose significant threats to IoT devices that run outdated firmware versions. Consequently, accurately and efficiently identifying firmware versions of devices is crucial to detecting device vulnerabilities and enhancing the overall security of IoT ecosystems.

In this work, we present DeepFW, the first Deep Neural Network (DNN)-based and fine-grained firmware version identification framework. DeepFW employs a Fusion Feature Attention Network (FFAN) to extract subtle differences in file systems within firmware, facilitating the identification of firmware versions in online devices. Additionally, to address the challenge of high similarity between versions caused by firmware homogeneity in the supply chain, we propose a novel metric loss, namely the Hard Mining Cosine Triplet-Center Loss (HCTCL), to enhance intra-class compactness and inter-class separability. To validate the effectiveness of our method, we collected 4,442 firmware images and obtained 130,445 valid embedded web interface files. Experimental results show that DeepFW significantly outperforms current fingerprinting schemes in terms of accuracy and efficiency in firmware version identification, achieving an accuracy rate of 96.83%. Using DeepFW, we identified 6,684 IoT devices on the internet that are vulnerable due to outdated firmware versions. Our evaluation indicates that only 2.28% of the IoT devices are running the latest firmware, with an average vulnerability rate of 61.26%.

Due to the sensitivity of data sourced from firmware or online devices, we provide only simplified pretrained models and code in the repository to facilitate the reproduction of our work.

## Installation

1. Clone the repository：
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a version of PyTorch with CUDA support installed if using a GPU.

## Usage Instructions

### Data Preparation

1. The dataset is stored in `dataset_demo`, containing embedded web pages from online IoT devices, from six firmware versions of four manufacturers: D-Link, TP-LINK, Linksys, and Netgear.
2. The `data_utils.py` file contains some utility functions and classes for data preprocessing and loading. These functionalities are mainly used for processing and preparing text data, especially for using pre-trained word vector models (such as GloVe) and custom datasets.
### Model Definition

1. Model definitions are located in the `models` directory. Key files include:
-  `FFAN.py`: Includes multi-scale convolutional layers, self-attention layers, and custom state updating layers for feature extraction.
-  `classifier.py`: Fully connected layers + softmax for final classification.
2. Loss functions are defined in the `losses` directory, with a key file:
- `HCTCL.py`: Custom Hard Mining Cosine Triplet-Center Loss for optimizing model training.
### Training the Model

Configure model parameters and training settings, such as learning rate, batch size, and the number of training epochs in `train.py`. Run the script to train the model.

```bash
python train.py
```

After training, the model weights will be saved.
### Testing the Model

Use `test.py` to test the trained model. This module loads the saved model weights and evaluates model performance.
```bash
python test.py
```
The test results include the classification accuracy of the model.

## Notes

- Ensure consistent `device` settings across all scripts to avoid tensor mismatch issues between GPU/CPU.
- Data file paths should be correctly set in the code.
- When using a GPU for training and testing, ensure CUDA is available and properly configured.
