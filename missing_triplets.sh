G=triplet_tcm_cv_2026-10-20_17-31-55
CUDA_VISIBLE_DEVICES=1 python main.py --task triplet_recognition --fold 3 --checkpoint outputs/2026-10-19_20-18-17_triplet_recognition_fold3/best.pt --temporal_model tcm --wandb_group $G
CUDA_VISIBLE_DEVICES=1 python main.py --task triplet_recognition --fold 4 --checkpoint outputs/2026-10-19_20-51-20_triplet_recognition_fold4/best.pt --temporal_model tcm --wandb_group $G
CUDA_VISIBLE_DEVICES=1 python main.py --task triplet_recognition --fold 5 --checkpoint outputs/2026-10-19_21-24-46_triplet_recognition_fold5/best.pt --temporal_model tcm --wandb_group $G