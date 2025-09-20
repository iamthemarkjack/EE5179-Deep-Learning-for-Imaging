## Assignment - 1 

### Directory Structure 

```
.
├── torch_model.py        # Defines the MLP using PyTorch
├── torch_train.py        # Training + evaluation for PyTorch model
├── custom_model.py       # Defines the custom MLP
├── custom_train.py       # Training + evaluation for custom model
├── data/                 # MNIST dataset (Will be downloaded during the first execution)
├── custom_model_exps/
│   ├── confmatrix_relu_lambda0p001.png
│   ├── confmatrix_sigmoid_lambda0.png
│   └── ... (all results for the custom model are saved here)
├── torch_model_exps/
│   ├── confmatrix_relu_lambda0p001.png
│   ├── confmatrix_sigmoid_lambda0.png
│   └── ... (all results for the torch model are saved here)
├── sample_predictions.png # Shows the sample predictions from the custom model
└── README.md             # This file
```

---

### Requirements

```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```

---

