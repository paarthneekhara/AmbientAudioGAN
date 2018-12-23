export CUDA_VISIBLE_DEVICES="1"
TRAIN_DIR=/data2/paarth/TrainDir/debugAmbient
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/chris/data/sc09/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64" \
	--train_summary_every_nsecs 30

export CUDA_VISIBLE_DEVICES="0"
TRAIN_DIR=/data2/paarth/TrainDir/debugAmbient500
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/chris/data/sc09/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=500.0" \
	--train_summary_every_nsecs 30

 python3 train_evaluate_long.py train ${TRAIN_DIR} --data_dir /data2/chris/data/timit_riff_trans/train --data_fastwav --model_overrides "objective=l1,batchnorm=False,subseq_len=65536,train_batch_size=16" --train_summary_every_nsecs 30 --ae_ckpt_fp /data2/paarth/TrainDir/WaveAE/WaveAEsc09_l1batchnormFalse/eval_sc09_valid/best_valid_l2-253802


export CUDA_VISIBLE_DEVICES="0"
TRAIN_DIR=/data2/paarth/TrainDir/debugAmbientPiano100
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/tatum/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0" \
	--train_summary_every_nsecs 30 \
	--data_randomize_offset


 python3 train_evaluate_long.py train ${TRAIN_DIR} --data_dir /data2/chris/data/timit_riff_trans/train --data_fastwav --model_overrides "objective=l1,batchnorm=False,subseq_len=65536,train_batch_size=16" --train_summary_every_nsecs 30 --ae_ckpt_fp /data2/paarth/TrainDir/WaveAE/WaveAEsc09_l1batchnormFalse/eval_sc09_valid/best_valid_l2-253802

TRAIN_DIR=Data/train/debug
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir Data/debug \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=2,alpha=100.0" \
	--train_summary_every_nsecs 30 \
	--data_overlap_ratio 0.5 \
	--train_ckpt_every_nsecs 60 \
	--data_randomize_offset

TRAIN_DIR=Data/train/debug
python3 train_evaluate.py eval \
	${TRAIN_DIR} \
	--data_dir Data/val \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=2,alpha=100.0" \
	--train_summary_every_nsecs 30 \
	--data_overlap_ratio 0.5 \
	--data_randomize_offset