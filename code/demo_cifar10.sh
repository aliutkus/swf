# parameters for sketching
NUM_QUANTILES=100
NUM_THETAS=500
CLIPTO=5000

# parameters for SWF
STEPSIZE=500
REG=0.000001
INPUT_DIM=100
BATCHSIZE=1


if [ $1 = "batch" ]; then
  # batch version
  echo "Processing CIFAR10 in the batch mode"

  NUM_SKETCHES=500

  python qsketch/sketch.py CIFAR10 --num_sketches $NUM_SKETCHES --num_quantiles $NUM_QUANTILES --clip $CLIPTO --num_thetas $NUM_THETAS --output ~/tmp/sketchCIFAR10 --projectors RandomProjectors --memory_usage 2 --root_data_dir qsketch/data/CIFAR10
  python batchIDT.py ~/tmp/sketchCIFAR10.npy  --input_dim $INPUT_DIM --num_samples 3000  --plot --stepsize $STEPSIZE --reg $REG --batchsize $BATCHSIZE --epochs 100 --plot_dir samples_cifar10_batch --logdir logs/cifar10/batch
fi

if [ $1 = "stream" ]; then
  echo "Processing CIFAR10 in the stream mode"

  # streaming version
  python streamIDT.py CIFAR10  --memory_usage 2 --root_data_dir qsketch/data/CIFAR10  --clip $CLIPTO --num_quantiles $NUM_QUANTILES  --projectors RandomProjectors  --input_dim $INPUT_DIM --num_samples 3000 --plot_target toydata.npy --plot --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS --batchsize $BATCHSIZE --plot_dir samples_cifar10_stream --logdir logs/cifar10/stream
fi
