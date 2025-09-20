import numpy as np

class MLP:
    def __init__(self, input_dim=784, hidden_dims=(500, 250, 100), output_dim=10, 
                 activations=("relu", "relu", "relu"), lr=1e-2, l2_lambda=0.0, seed=None):

        if seed is not None:
            np.random.seed(seed)

        sizes = [input_dim] + list(hidden_dims) + [output_dim]
        self.num_layers = len(sizes) - 1

        assert len(activations) == len(hidden_dims)
        self.activations = activations

        # Weight initialization
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            Ni, No = sizes[i], sizes[i+1]
            M = np.sqrt(6 / (Ni + No))
            self.W.append(np.random.uniform(-M, M, size=(Ni, No)))
            self.b.append(np.zeros(No))

        self.lr = lr
        self.l2_lambda = l2_lambda
        self.zs = None
        self.acts = None
        self.last_loss = None

    # Activation functions and their derivatives
    @staticmethod
    def relu(z): return np.maximum(0, z)
    @staticmethod
    def relu_grad(z): return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    @staticmethod
    def sigmoid_grad(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    @staticmethod
    def tanh(z): return np.tanh(z)
    @staticmethod
    def tanh_grad(z): return 1 - np.tanh(z)**2

    
    @staticmethod
    def softmax(z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z_stable)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def compute_loss(self, y_hat, y):
        N = y.shape[0]
        eps = 1e-12
        correct_logprobs = -np.log(np.clip(y_hat[np.arange(N), y], eps, None))
        loss = np.mean(correct_logprobs)

        if self.l2_lambda > 0:
            l2_term = 0.5 * self.l2_lambda * sum(np.sum(W**2) for W in self.W)
            loss += l2_term / N

        return loss

    def _apply_activation(self, z, name):
        if name == "relu": return self.relu(z)
        if name == "sigmoid": return self.sigmoid(z)
        if name == "tanh": return self.tanh(z)
        raise ValueError(f"Unknown activation: {name}")

    def _apply_activation_grad(self, z, name):
        if name == "relu": return self.relu_grad(z)
        if name == "sigmoid": return self.sigmoid_grad(z)
        if name == "tanh": return self.tanh_grad(z)
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, X):
        a = X
        acts = [a]
        zs = []

        for i in range(self.num_layers):
            z = a.dot(self.W[i]) + self.b[i]
            zs.append(z)
            if i == self.num_layers - 1:
                a = self.softmax(z)
            else:
                a = self._apply_activation(z, self.activations[i])
            acts.append(a)

        self.zs = zs
        self.acts = acts
        return acts[-1]

    def backward(self, X, y):
        N = y.shape[0]
        C = self.b[-1].shape[0]

        y_onehot = np.zeros((N, C))
        y_onehot[np.arange(N), y] = 1.0

        dW = [None] * self.num_layers
        db = [None] * self.num_layers

        y_hat = self.acts[-1]
        delta = (y_hat - y_onehot) / N

        for l in reversed(range(self.num_layers)):
            a_prev = self.acts[l]
            dW[l] = a_prev.T.dot(delta)
            db[l] = np.sum(delta, axis=0)

            if self.l2_lambda > 0:
                dW[l] += (self.l2_lambda / N) * self.W[l]

            if l > 0:
                z_prev = self.zs[l-1]
                act_name = self.activations[l-1]
                delta = (delta.dot(self.W[l].T)) * self._apply_activation_grad(z_prev, act_name)

        return {"dW": dW, "db": db}

    def update_params(self, grads):
        for i in range(self.num_layers):
            self.W[i] -= self.lr * grads["dW"][i]
            self.b[i] -= self.lr * grads["db"][i]

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)