# Paper-defect-type-classification
1. Train
 - scripts train example with model convnext_xlarge_22k_224.pth and fold_2:
    python main.py --epochs 12 --model convnext_xlarge --model_ema True --batch_size 32 --input_size 224  --data_set IMNET --data_path E:\data_augmentation_ver_2\fold_data\fold_2  --
nb_classes 19 --drop_path 0.2 --num_workers 8 --warmup_epochs 4 --save_ckpt true --output_dir checkpoints/convnext_large_fold2_224_bs_32 --finetune weights/convnext_xlarge_22k_224.pth --cutmix 0 --mixup 0.2 --hflip 0.5 --vflip 0.5
  --color_jitter 0.1  --lr 1e-4  --min_lr 1e-6  --project convnext_large_fold1_224_bs_32_ema --seed 69

 - scripts train example with model convnext_base_22k_224.pth and fold_5:
    python main.py --epochs 36 --model convnext_base --model_ema True --batch_size 32 --input_size 224  --data_set IMNET --data_path  E:\data_augmentation_ver_2\fold_data\fold_2 --nb_classes 1
    9 --drop_path 0.2 --num_workers 8 --warmup_epochs 4 --save_ckpt true --output_dir checkpoints/convnext_base_fold0_224_bs_32 --finetune weights/convnext_base_22k_224.pth --cutmix 0 --mixup 0.2 --hflip 0.5 --vflip 0.5  --color_jitte
    r 0.2  --lr 1e-4  --min_lr 1e-6  --project convnext_large_fold1_224_bs_32_ema --seed 69
2. Predict
   
    a. download model from:
    https://drive.google.com/file/d/1U2TALM0fwr0dhdK68EgUEkcpv1v6phnQ/view?usp=sharing
    
    b. extract to ./checkpoints

    c. python predict.py