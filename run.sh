PRETRAINED_CKPT_PATH=checkpoint/uncased_L-24_H-1024_A-16/params
DATA_PATH=xnli
bert_config_path=checkpoint/uncased_L-24_H-1024_A-16/bert_config.json
vocab_path=checkpoint/uncased_L-24_H-1024_A-16/vocab.txt

sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH 1> logs/time0
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH 1> logs/time1
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH 1> logs/time2
sh train_cls.sh $PRETRAINED_CKPT_PATH $bert_config_path $vocab_path $DATA_PATH 1> logs/time4
