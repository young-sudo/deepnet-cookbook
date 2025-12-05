# A Cookbook for Deep Neural Networks

*by Younginn Park*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003366?style=for-the-badge&logo=plotly&logoColor=white)
![ClearML](https://img.shields.io/badge/ClearML-00BFA6?style=for-the-badge&logo=mlflow&logoColor=white) 

Repository contains solutions for selected Deep Neural Network architecture optimization tasks using **PyTorch** and fine-tuned for running in the Google Colab environment.

The implemented solutions were a part of the Deep Neural Network course at the University of Warsaw 2024/25, where they achieved a **top 10% class ranking** for performance.

Projects include methods, proposed in the latest publications, with a potential for improving existing models:

## 1. **Proximal Backpropagation ([ProxyProp](https://arxiv.org/abs/1706.04638))**

Modification for the backpropagation algorithm by taking implicit instead of explicit gradient steps to update the network parameters during neural network training.

The update of weights is defined as:

$$
%\begin{equation}
W^{(l)} = \text{arg min}_{W} \frac{1}{2} || W \cdot g^{(l-1)} + b^{(l)} - f^{(l)}_{*} ||^2 + \frac{1}{2\eta} || W - W^{(l)} ||^2
%\end{equation}
$$

## 2. **Adversarial Training ([AdvProp](https://arxiv.org/abs/1911.09665) and [SparseTopK](https://openreview.net/forum?id=QzcZb3fWmW))**

Improvement of robustness for image classification models by training on batches containing both clean and adversarially perturbed examples. This forces the model to learn features that are invariant to small, malicious input changes, leading to better generalization and resilience against attacks.

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/cats.png" alt="cats" width="500"/>
  </figure>
</div>

## 3. **Differential Attention ([Diff Attention](https://arxiv.org/abs/2410.05258))**

Enhancement for transformer attention by allowing the model to amplify attention to the relevant context while canceling noise by calculating attention scores as the difference between two separate softmax attention maps.

Differential Attention works similarly to normal Attention, but with a few differences:
* Differential Attention partitions $Q$ and $K$ matrices into chunks: $Q_1$, $Q_2$, $K_1$, $K_2$. Each of them have a shape `(b, l, n_h, head / 2)`
* It calculates two attention matrices: $A_1$ and $A_2$ using respective $Q_i, K_i$ matrices. Note that here the attention pre-activation is normalized by $ \sqrt{head / 2}$. Causal mask is still added to the pre-activation.
* The final attention matrix is calculated as $A = A_1 - \lambda A_2$
* $\lambda$ in the equation above is calculated as $\lambda = \exp(\lambda_{K_1} * \lambda_{Q_1}) - \exp(\lambda_{K_2} * \lambda_{Q_2}) + \lambda_{\text{init}}$, where $\lambda_{K_1}, \lambda_{Q_1}, \lambda_{K_2}, \lambda_{Q_2}$ are all parameters of shape `head / 2`, $*$ denotes scalar multiplication of vectors and $\lambda_{\text{init}} = 0.8$
* Additionally, normalization is applied to per-head tokens $O' = \text{RMSNorm}(O') \cdot (1-\lambda_{init})$

## 4. **Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) and Random Network Distillation ([RND](https://arxiv.org/abs/1810.12894))**

Improving exploration in reinforcement learning, enabling agents to learn more effectively in sparse reward environments.

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/deepnet-cookbook/main/img/nethack.png" alt="nethack" width="500"/>
    <br>
    <figcaption style="text-align:center;"><em>Roguelike game NetHack has been used to train the Agent</em></figcaption>
  </figure>
</div>
