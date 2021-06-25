# Adversarial Attack on CIFAR-10 Dataset using PyTorch

- I implemented **FGSM**(Fast Gradient Sign Method), **PGD**(Projected Gradient Descent), **SD**(Speedest Descent), [**DeepFool**](https://arxiv.org/abs/1511.04599) and [**C&W**](https://arxiv.org/abs/1608.04644) on CIFAR-10 dataset.
- The implementations is about untargeted attack for brevity, but can be easily moved into targeted attack with a little effort. 
- The base model used in this project is the pretrained mobilenet_v2 provided by [huyvnphan](https://github.com/huyvnphan/PyTorch_CIFAR10).

## Result

| Type | Accurary | Mean of Distortion |
| ---- | ---- | ---- |
| None | 93.91% | 0 |
| FGSM | 71.27% | -3.29e-03 |
|   PGD   | 74.12% | -5.76e-03 |
| SD_linf | 73.88% | -5.19e-03 |
| SD_l2 | 57.83% | 5.25e-02 |
| DeepFool | 7.07% | -2.47e-02 |
| C&W | 86.28% | 4.77e-03 |

Note that DeepFool got extremely successful attack rate, but most of distorted images are perceptible.
