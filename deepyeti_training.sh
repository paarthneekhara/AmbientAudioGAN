export CUDA_VISIBLE_DEVICES="0"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/tatum_dp_standardAE_new
# rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=150.0,enc_length=64,use_skip=False" \
	--train_summary_every_nsecs 60 

export CUDA_VISIBLE_DEVICES="-1"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/tatum_dp_standardAE_new
python3 train_evaluate.py eval \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/valid \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=2,alpha=150.0,enc_length=64,use_skip=False"

export CUDA_VISIBLE_DEVICES="-1"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/tatum_dp_standardAE_new
python3 train_evaluate.py infer \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/valid8s \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=1,alpha=150.0,enc_length=64,use_skip=False,subseq_len=131072"


export CUDA_VISIBLE_DEVICES="1"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/tatum_dp_skip_16
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0,enc_length=16,use_skip=True,stride=4,skip_limit=3" \
	--train_summary_every_nsecs 60 

export CUDA_VISIBLE_DEVICES="-1"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/tatum_dp_skip_16
python3 train_evaluate.py eval \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/valid \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=2,alpha=100.0,enc_length=16,use_skip=True,stride=4,skip_limit=3"


export CUDA_VISIBLE_DEVICES="1"
TRAIN_DIR=/datasets/home/10/610/pneekhar/TRAIN/Ambient/sc09_dp_noskip
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /datasets/home/10/610/pneekhar/DATA/clipped_dp_256_04/sc09/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0,use_skip=False,enc_length=64" \
	--train_summary_every_nsecs 60

export CUDA_VISIBLE_DEVICES="1"
TRAIN_DIR=/datasets/home/10/610/pneekhar/TRAIN/Ambient/sc09_dp_noskip
python3 train_evaluate.py eval \
	${TRAIN_DIR} \
	--data_dir /datasets/home/10/610/pneekhar/DATA/clipped_dp_256_04/sc09/valid \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0,use_skip=False,enc_length=64"



export CUDA_VISIBLE_DEVICES="2"
TRAIN_DIR=/datasets/home/10/610/pneekhar/TRAIN/Ambient/sc09_dp_SEGAN_arch
rm -rf ${TRAIN_DIR}
python3 train_evaluate.py train \
	${TRAIN_DIR} \
	--data_dir /datasets/home/10/610/pneekhar/DATA/clipped_dp_256_04/sc09/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0,enc_length=16,stride=2,kernel_len=31" \
	--train_summary_every_nsecs 60

export CUDA_VISIBLE_DEVICES="2"
TRAIN_DIR=/datasets/home/10/610/pneekhar/TRAIN/Ambient/sc09_dp_SEGAN_arch
python3 train_evaluate.py eval \
	${TRAIN_DIR} \
	--data_dir /datasets/home/10/610/pneekhar/DATA/clipped_dp_256_04/sc09/valid \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=100.0,enc_length=16,stride=2,kernel_len=31"




export CUDA_VISIBLE_DEVICES="0"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/DC_tatum_dp_standardAE
rm -rf ${TRAIN_DIR}
python3 train_declipper.py train \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/train \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=64,alpha=200.0,enc_length=64,use_skip=False" \
	--train_summary_every_nsecs 60 

export CUDA_VISIBLE_DEVICES="-1"
TRAIN_DIR=/data2/paarth/TrainDir/Ambient/DC_tatum_dp_standardAE
python3 train_declipper.py eval \
	${TRAIN_DIR} \
	--data_dir /data2/paarth/ambient/clipped_dp_512_04/tatum/valid \
	--data_fastwav \
	--model_overrides "objective=l1,batchnorm=False,train_batch_size=2,alpha=200.0,enc_length=64,use_skip=False"