alias python="/Applications/anaconda3/python.app/Contents/MacOS/python"
export PATH="/Applications/anaconda3/bin:$PATH"
export BERT_BASE_DIR=/Users/yangxuhang/Downloads/uncased_L-12_H-768_A-12
export MY_DATASET='/Users/yangxuhang/PycharmProjects/MicrosoftPTA/bert-NewsClassifier/dataset'

python '/Users/yangxuhang/PycharmProjects/MicrosoftPTA/bert-NewsClassifier/run_classifier.py' \
  --task_name=news \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=16 \
  --train_batch_size=250 \
  --learning_rate=5e-5 \
  --num_train_epochs=5.0 \
  --output_dir=/Users/yangxuhang/Desktop/bertmodel