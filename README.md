# SSFE-Net
Code for "SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning"

## Abstract
Ultra-Fine-Grained Visual Categorization (ultra-FGVC) has become a popular problem due to its great real-world potential for classifying the same or closely related species with very similar layouts. However, there present many challenges for the existing ultra-FGVC methods, firstly there are always not enough samples in the existing ultra-FGVC datasets based on which the models can easily get overfitting. Secondly, in practice, we are likely to find new species that we have not seen before and need to add them to existing models, which is known as incremental learning. The existing methods solve these problems by Few-Shot Class Incremental Learning (FSCIL), but the main challenge of the FSCIL models on ultra-FGVC tasks lies in their inferior discrimination detection ability since they usually use low-capacity networks to extract features, which leads to insufficient discriminative details extraction from ultra-fine-grained images. In this paper, a self-supervised feature enhancement for the few-shot incremental learning network (SSFE-Net) is proposed to solve this problem. Specifically, a self-supervised learning (SSL) and knowledge distillation (KD) framework is developed to enhance the feature extraction of the low-capacity backbone network for ultra-FGVC few-shot class incremental learning tasks. Besides, we for the first time create a series of benchmarks for FSCIL tasks on two public ultra-FGVC datasets and three normal fine-grained datasets, which will facilitate the development of the Ultra-FGVC community. Extensive experimental results on public ultra-FGVC datasets and other state-of-the-art benchmarks consistently demonstrate the effectiveness of the proposed method.

## Dataset
We provide the source code on two benchmark datasets at the moment, i.e., CUB200 and miniImageNet. The source code for the Ultra-FGVC part of the training will be updated shortly. For the existing two datasets, please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [CSS](https://github.com/anyuexuan/CSS))

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
