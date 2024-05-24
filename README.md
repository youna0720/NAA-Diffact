# Action anticipation using Tranformer

### Environment :
* We use Anaconda3 env for this project, and you can install environments using environment.yaml file.
* We use 4 gpus for training and evaluation, and 1 gpu for prediction. 

### Training:

* download the data from https://mega.nz/file/O6wXlSTS#wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8
* extract it so that you have the `data/breakfast` folder in the same directory as `main.py` './breakfast'.
* To change the default saving directory or the model parameters, check the list of options by running `./scripts/train.sh local $runs_num` (runs_num indicates your run number).
*  argument file : opts.py
   `--anticipate` : future anticipation
   `--input_type` : i3d_transcript - I3D feature, gt - GT lable for the encoder input

### Evaluation:
* Evaluation code is to evaluate models with in two way - Autoregressive, and oneshot.
* To check the list of options run `./scripts/evaluate.sh local `.
* mode == 'ar' : auto-regressive inference
* mode == 'oneshot' : gt input for transformer decoder

### Prediction:
* Prediction code is for testset.
* It finally calculates MoC results of the models.
* To check the list of options run `./scripts/predich.sh local `.
* MoC results
