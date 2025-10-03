## Assignment 2 - Convolutional Neural Networks

---

### Directory Structure

- `model.py` : Defines the CNN model (with optional BatchNorm).
- `utils.py` : Dataset loaders and plotting utilities.
- `train.py` : Training & evaluation script.
- `visualize.py` : Tools to visualize filters, activations, occlusion maps.
- `adversarial.py` : Scripts to generate adversarial examples (non-targeted & targeted).
- `requirements.txt` : Required Python packages.

---

### How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the CNN **without BatchNorm**:
   ```bash
   python3 train.py --epochs 10
   ```

3. Train the CNN **with BatchNorm**:
   ```bash
   python3 train.py --epochs 10 --use_batchnorm
   ```

4. After training, results will be saved in `./checkpoints`:
   - `mnist_cnn_best.pt` : best model weights
   - `losses.png` : training/validation loss curves
   - `preds.png` : sample test predictions

5. Use the saved checkpoint to run the following files:
   - Use `visualize.py` to plot learned filters, feature maps, and occlusion maps.
   - Use `adversarial.py` to generate adversarial examples.
   - The results of the above and the comparison between with batchnorm and without batchnorm is stored `./images`.
---

### 1. Dimensions of Input and Output at Each Layer
- Input: **1×28×28**
- Conv1: 32×28×28
- Pool1: 32×14×14
- Conv2: 32×14×14
- Pool2: 32×7×7
- Flatten: 1568
- FC1: 500
- FC2: 10 (class logits)

### 2. Number of Parameters
- Conv1: (3×3×1×32) + 32 = **320**
- Conv2: (3×3×32×32) + 32 = **9,248**
- FC1: (1568×500) + 500 = **784,500**
- FC2: (500×10) + 10 = **5,010**

**Total Parameters = 799,078**
- Conv layers: 9,568
- FC layers: 789,510

### 3. Number of Neurons (activations during forward pass)
- Conv1: 32×28×28 = **25,088**
- Pool1: 32×14×14 = **6,272**
- Conv2: 32×14×14 = **6,272**
- Pool2: 32×7×7 = **1,568**
- FC1: **500**
- FC2: **10**

**Total Neurons = 39,710**
- Conv layers: 39,200
- FC layers: 510

### 4. Effect of Batch Normalization
- Adding `BatchNorm2d` after conv layers:
  - Not much improvement in **validation/test accuracy** for this dataset.
  - But in complex scenarios may help with gradient issues.