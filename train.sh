python train.py -project ssfe -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 -lr_base 0.002 \
-lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 20 60 80 -gpu 0 \
-temperature 16 -dataroot ./data -batch_size_base 32 -from_scratch

python train.py -project ssfe -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 \
-lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Cosine -gpu 0 -temperature 16 -dataroot ./data -from_scratch

python train.py -project ssfe -dataset PlantVillage -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 \
-lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Milestone -milestones 60 90 -gpu 0 \
-temperature 16 -dataroot ./data -batch_size_base 32 -from_scratch
