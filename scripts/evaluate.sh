#if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
#    cd $PBS_O_WORKDIR
#fi
echo "split1"
python main_next.py --eval --next --seg --pos_emb --runs=$2 --model=detr --mode=train --input_type=i3d_transcript --split=1 \
    --dataset=ek55 --input_seq_len 1
