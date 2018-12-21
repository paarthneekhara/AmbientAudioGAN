export CUDA_VISIBLE_DEVICES="1"
TRAIN_DIR=/data2/paarth/TrainDir/debugAmbient
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/chris/data/sc09/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False" \
	--train_summary_every_nsecs 30 \