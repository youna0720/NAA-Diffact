CUDA_VISIBLE_DEVICES=0 python main_next.py --wandb 'DETR__, batch_size=32, 512, enc2, dec2, head=4, transformer, dropout0.8_feat, feat_loss, weight_decay=1e-2, f=8, qk_attn/ runs115'\
    --feat_loss --qk_attn --next --n_encoder_layer 6 --n_decoder_layer 6 --nhead 8 --frame 8 \
    --lr 1e-5 --hidden_dim 512 --runs=$1 --batch_size 32 --epochs 40 --weight_decay 1e-2\
    --model=detr --mode=train --split=1 --dataset=ek55 \
