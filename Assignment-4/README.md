
EE5179 PA04 - Recurrent Neural Networks
Files:
- utils.py : shared RNN/LSTM/GRU model class
- mnist_rnn_vanilla.py : vanilla RNN for MNIST
- mnist_rnn_lstm.py : LSTM for MNIST
- mnist_rnn_bidir.py : Bidirectional LSTM for MNIST
- binary_addition_rnn.py : LSTM for adding binary strings

Usage examples:
- Train vanilla RNN on MNIST:
    python mnist_rnn_vanilla.py --epochs 10 --batch-size 128 --hidden-size 128
- Train LSTM on MNIST:
    python mnist_rnn_lstm.py --epochs 10
- Train bidirectional LSTM on MNIST:
    python mnist_rnn_bidir.py --epochs 10
- Train binary addition RNN:
    python binary_addition_rnn.py --seq-len 8 --hidden-size 16 --epochs 20 --loss bce

Notes:
- Scripts automatically download MNIST into ./data
- Output models and plots saved in outputs_* directories
- For custom handwritten images, load and preprocess to 28x28 grayscale and pass through the model code.
