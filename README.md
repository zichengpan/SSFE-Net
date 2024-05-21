# [WACV 2023] SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning
Code for "SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning", [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Pan_SSFE-Net_Self-Supervised_Feature_Enhancement_for_Ultra-Fine-Grained_Few-Shot_Class_Incremental_Learning_WACV_2023_paper.html) (WACV23)

## Abstract
Ultra-Fine-Grained Visual Categorization (ultra-FGVC) has become a popular problem due to its great real-world potential for classifying the same or closely related species with very similar layouts. However, there present many challenges for the existing ultra-FGVC methods, firstly there are always not enough samples in the existing ultra-FGVC datasets based on which the models can easily get overfitting. Secondly, in practice, we are likely to find new species that we have not seen before and need to add them to existing models, which is known as incremental learning. The existing methods solve these problems by Few-Shot Class Incremental Learning (FSCIL), but the main challenge of the FSCIL models on ultra-FGVC tasks lies in their inferior discrimination detection ability since they usually use low-capacity networks to extract features, which leads to insufficient discriminative details extraction from ultra-fine-grained images. In this paper, a self-supervised feature enhancement for the few-shot incremental learning network (SSFE-Net) is proposed to solve this problem. Specifically, a self-supervised learning (SSL) and knowledge distillation (KD) framework is developed to enhance the feature extraction of the low-capacity backbone network for ultra-FGVC few-shot class incremental learning tasks. Besides, we for the first time create a series of benchmarks for FSCIL tasks on two public ultra-FGVC datasets and three normal fine-grained datasets, which will facilitate the development of the Ultra-FGVC community. Extensive experimental results on public ultra-FGVC datasets and other state-of-the-art benchmarks consistently demonstrate the effectiveness of the proposed method.

## Dataset
We provide the source code on two benchmark datasets at the moment, i.e., CUB200 and miniImageNet. The source code for the Ultra-FGVC part of the training will be updated shortly. For the existing two datasets, please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.


## Training Scripts
You can run the code by the following commands or simply use ```sh train.sh```.
- Train CUB200
    ```
    python train.py -project ssfe -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 20 60 80 -gpu '0' -temperature 16 -dataroot ./data -batch_size_base 32 -from_scratch
    ```
- Train Mini-ImageNet
    ```
    python train.py -project ssfe -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0 -temperature 16 -dataroot ./data -from_scratch
    ```

## Citation
If you find our code or paper useful, please give us a citation.
```bash
@InProceedings{Pan_2023_WACV,
    author    = {Pan, Zicheng and Yu, Xiaohan and Zhang, Miaohua and Gao, Yongsheng},
    title     = {SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {6275-6284}
}
```

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [CSS](https://github.com/anyuexuan/CSS)

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
