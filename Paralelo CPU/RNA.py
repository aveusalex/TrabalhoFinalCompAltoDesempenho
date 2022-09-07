import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F


def normaliza(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    x_trans = scaler.transform(x)

    return x_trans


def flat(x):
    with torch.no_grad():
        # Tornando a imagem que é de duas dimensões em apenas uma.
        flatten = nn.Flatten()
        flat_image_ = flatten(x)
    return flat_image_


def transforma_y(x, y):
    y_train_adequado = torch.zeros((x.shape[0], 10)).float()  # sempre são 10 classes nesse caso
    aux = 0
    for rotulo in y.tolist():
        y_train_adequado[aux, int(rotulo)] = 1
        aux += 1  # apenas um contador que organiza as linhas

    return y_train_adequado


# rede 2 do artigo do yan lecun
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(28*28, 12),
            nn.Tanh(),
            nn.Linear(12, 10)
        )
        self.weights = torch.cat((torch.flatten(self.linear_tanh_stack[0].weight),
                                  torch.flatten(self.linear_tanh_stack[0].bias),
                                  torch.flatten(self.linear_tanh_stack[2].weight),
                                  torch.flatten(self.linear_tanh_stack[2].bias)))
        self.training_data = datasets.MNIST(
            root=r"F:\Mnist",
            train=True,
            download=True,
            transform=ToTensor()
        )
        self.test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        self.loss_function = F.mse_loss
        self.X, self.y = self.ajeita_dados()

    def forward(self, x):
        mnist = self.linear_tanh_stack(x)
        mnist_soft = self.softmax(mnist)
        return mnist_soft

    def pass_weights_to_net(self):
        camada1_size = self.linear_tanh_stack[0].weight.shape[0] * self.linear_tanh_stack[0].weight.shape[1]
        camada1_bias_size = self.linear_tanh_stack[0].bias.shape[0]
        camada2_size = self.linear_tanh_stack[2].weight.shape[0] * self.linear_tanh_stack[2].weight.shape[1]
        self.linear_tanh_stack[0].weight = nn.Parameter(torch.reshape(torch.tensor(self.weights[:camada1_size]), self.linear_tanh_stack[0].weight.shape))
        self.linear_tanh_stack[0].bias = nn.Parameter(torch.reshape(torch.tensor(self.weights[camada1_size:camada1_size+camada1_bias_size]), self.linear_tanh_stack[0].bias.shape))
        self.linear_tanh_stack[2].weight = nn.Parameter(torch.reshape(torch.tensor(self.weights[camada1_size+camada1_bias_size:camada1_size+camada1_bias_size+camada2_size]), self.linear_tanh_stack[2].weight.shape))
        self.linear_tanh_stack[2].bias = nn.Parameter(torch.reshape(torch.tensor(self.weights[camada1_size+camada1_bias_size+camada2_size:]), self.linear_tanh_stack[2].bias.shape))

    def ajeita_dados(self):
        qtd = 500
        X = self.training_data.data[:qtd, :, :]
        y = self.training_data.targets[:qtd]
        del self.training_data
        del self.test_data
        X = normaliza(flat(X))
        y = transforma_y(X, y)
        X = torch.from_numpy(X)
        return X, y

    def run(self):
        with torch.no_grad():
            y_pred = self.forward(self.X)
            loss = self.loss_function(y_pred, self.y)
        return loss.numpy()


if __name__ == "__main__":
    net = NeuralNetwork()
    loss = net.run().item()
    print(loss)
