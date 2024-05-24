import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", default="transformer", help="select model: [\"rnn\", \"cnn\", \
                    \"transformer\",\"detr\",\"avth\"]")
parser.add_argument("--mode", default="train_eval", help="select action: [\"train\", \
                    \"predict\", \"train_eval\"]")
parser.add_argument("--dataset", default='breakfast', help="breakfast, ek55")
parser.add_argument('--eval', "-e", action='store_true', help="evaluation mode")
parser.add_argument('--predict', "-p", action='store_true', help="predict for whole videos mode")

#breakfast specific parameter
parser.add_argument("--mapping_file", default="./breakfast/mapping.txt")
parser.add_argument("--features_path", default="./breakfast/features/")
parser.add_argument("--gt_path", default="./breakfast/groundTruth/")
parser.add_argument("--split", default="1", help='split number')
parser.add_argument("--file_path", default="./breakfast/splits/")
parser.add_argument("--val_file_path", default="./breakfast/splits2/")
parser.add_argument("--model_save_path", default="./save_dir/models/transformer")
parser.add_argument("--results_save_path", default="./save_dir/results/transformer")

#Epic kitchen specific parameter
parser.add_argument("--frame", type=int, default=14)
parser.add_argument("--path_to_lmdb", default="./data/ek55/rgb")
parser.add_argument("--path_to_data", default="./data/ek55")
parser.add_argument("--time_step", type=float, default=0.25)
parser.add_argument("--img_tmpl", default='frame_{:010d}.jpg')
parser.add_argument("--Ta", type=float, default=1.0)
parser.add_argument("--feat_loss", action='store_true', help='feature reconstruction loss')
parser.add_argument("--feature_loss", type=bool, default=False)
parser.add_argument("--feature_one_loss", type=bool, default=False)
parser.add_argument("--feat_seg", action='store_true', help='feature segmentation')

parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--eval_epoch", type=int, default=30)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--workers", type=int, default= 10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr_mul", type=float, default=2.0)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("-warmup", '--n_warmup_steps', type=int, default=500)
parser.add_argument("--cpu", action='store_true', help='run in cpu')
parser.add_argument("--save_frea", type=int, default=1)
parser.add_argument("--sample_rate", type=int, default=3)
parser.add_argument("--obs_perc", default=0.2)
parser.add_argument("--max_action_seq", default=20)
parser.add_argument("--input_seq_len", default=1)

#CNN specific parameters
parser.add_argument("--n_segments", type=int, default=128)
parser.add_argument("--sigma", type=int, default=3, help="sigma for the gaussian smoothing step")

#Transformer specific parameters
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--n_encoder_layer", type=int, default=2)
parser.add_argument("--n_decoder_layer", type=int, default=2)
parser.add_argument("--n_avth_layer", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.8)
parser.add_argument("--avt_layer", type=int, default=12)

#Encoder specific parameters
parser.add_argument("--activity", action='store_true', help='activity classification')
parser.add_argument("--next", action='store_true', help='next action classification')
parser.add_argument("--next_dec", action='store_true', help='next action classification with decoder')
parser.add_argument("--seg", action='store_true', help='action segmentation')
parser.add_argument("--anticipate", action='store_true', help='future anticipation')
parser.add_argument("--sos", action='store_true', help='future anticipation')
parser.add_argument("--pos_emb", action='store_true', help='future anticipation')
parser.add_argument("--qk_attn", action='store_true', help='detr transformer')
parser.add_argument("--feat", action='store_true', help='feature loss')
parser.add_argument("--rel_pos_type", type=str, default='abs', help='relative pos emb type')
parser.add_argument("--rel_only", action='store_true', help='relative pos emb only')



#Test on GT or decoded input
parser.add_argument("--input_type", default="i3d_transcript", help="select input type: [\"decoded\", \"gt\"]")
parser.add_argument("--decoded_path", default="./data/decoded/split1")
parser.add_argument("--tf", default=False, help="export pre-trained weights from tensorflow")
parser.add_argument("--runs", default=0, help="save runs")
parser.add_argument("--epoch", default=40, help="eval_epoch")

parser.add_argument('--wandb', default='wandb',type=str, help='wandb name')
