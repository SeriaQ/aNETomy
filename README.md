<p align="center">
  <img src="https://ooo.0x0.ooo/2025/03/25/OS0SKi.png" alt="tcurve-icon.png" width = "168" height = "160" />
</p>
<h1 align="center">aNETomy</h1>

<p align="center">
  <b> A neat instrument designed for visualizing neural networks. </b><br>
  Experience seamless visualization without the burden of conversion.
</p>
<p align="center">
  <img src="https://img.shields.io/pypi/v/anetomy?color=blue&label=Version" alt="Version">
  <img src="https://img.shields.io/github/stars/SeriaQ/aNETomy?style=social" alt="GitHub Repo Stars">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>



------

## 🚀 Spotlight

🎯 **Effortless Visualization**

Given that PyTorch is an imperative framework, to visualize on the fly could be very useful for developers. aNETomy sidesteps the prep and conversion while keeping the accessibility and compatibility.

<p align="center">
  <img src="https://ooo.0x0.ooo/2025/03/26/OSYjrC.png" alt="graph.png" width = "280" height = "1288" />
</p>

------

## 📊 Comparison

The following comparison are concluded according to hands-on practical experience. It could be wrong somewhere.

| Feature                      | aNETomy | Netron | TensorBoard | torchviz | torchinfo | torchview | torchlens | torchexplorer |
| ---------------------------- | --------------- | --------- | ----------- | --------- | ------------ | --------- | --------- | ------------- |
| **Less Preparation** | ✅           | ❌    | ✅  | ✅ | ✅        | ✅ | ✅     | ✅    |
| **Handles Dynamic Models** | ✅     | ❌ | ⚠️   | ❌ | ⚠️       | ✅ | ✅ | ❌          |
| **Interactive Graphs**       | ✅ | ✅   | ✅  | ❌    | ❌         | ❌ | ❌    | ✅         |
| **Tensor Shapes** | ✅           | ✅     | ❌       | ✅     | ✅        | ✅     | ✅     | ✅         |
| **Structure Completeness** | ✅           | ✅ | ✅ | ❌ based on autograd | ⚠️ some functions are missing | ⚠️    | ✅ | ❌ based on autograd |
| **Compatibility**          | ✅           | ⚠️ unfriendly to some ops      | ⚠️ limited | ✅ | ❌ No         | ✅         | ✅     | ✅         |

------

## ⚡ Quick Start

You can simply take it as a wrapper or help you plot curves.

```python
from anetomy import Inspector
import torch
from torch import nn
import torch.nn.functional as F

# define the toy model
class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(28*28*2, 10)

    def forward(self, x):
        x = F.relu(x, inplace=True)
        x = x.reshape(-1,)
        y = self.lin(x)
        return y
          
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.mpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.dense = Dense()

    def forward(self, x, scale=1):
        x *= scale
        y = self.conv1(x)
        y = self.mpool1(y)
        y = self.dense(y)
        return y

# input dummy data to visualize
net = NeuralNetwork()
dummy_x = torch.randn(1, 3, 28, 28)
dummy_s = 2
ant = Inspector(graph_path='./toynn.png')
ant.dissect(net, dummy_x, scale=dummy_s)
ant.draw(3)
```



------

## 📦 Installation

At first, install graphviz beforehand.

### 🍎 macOS

Use Homebrew for a quick installation:

```bash
brew install graphviz
```

### 🐧 Linux

For Debian-based systems (Ubuntu, Debian):

```bash
sudo apt update && sudo apt install graphviz
```

For Arch-based systems (Arch, Manjaro):

```bash
sudo pacman -S graphviz
```

For RHEL-based systems (Fedora, CentOS):

```bash
sudo dnf install graphviz
```

### 🐍 Python

Then install the bindings.

```sh
pip install graphviz
```

Finally, install anetomy via pip.

```sh
pip install anetomy
```

------

## ❤️ Support

If you find aNETomy helpful, please give it a ⭐ on GitHub! ▶️ https://github.com/SeriaQ/aNETomy