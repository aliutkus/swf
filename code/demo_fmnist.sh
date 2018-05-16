# parameters for sketching
NUM_QUANTILES=200
NUM_THETAS=200
CLIPTO=5000

# parameters for SWF
STEPSIZE=300
REG=0
INPUT_DIM=100
BATCHSIZE=1

PROJECTORS="RandomLocalizedProjectors"

if [ $1 = "batch" ]; then
  # batch version
  echo "Processing FashionMNIST in the batch mode"

  NUM_SKETCHES=500

  python qsketch/sketch.py FashionMNIST --num_sketches $NUM_SKETCHES --num_quantiles $NUM_QUANTILES --clip $CLIPTO --num_thetas $NUM_THETAS --output ~/tmp/sketchFashionMNIST --projectors $PROJECTORS --memory_usage 2 --root_data_dir qsketch/data/FashionMNIST
  python batchIDT.py ~/tmp/sketchFashionMNIST.npy  --input_dim $INPUT_DIM --num_samples 3000  --plot --stepsize $STEPSIZE --reg $REG --batchsize $BATCHSIZE --epochs 100 --plot_dir samples_fmnist_batch_$PROJECTORS --log --logdir logs/fmnist/batch/$PROJECTORS
fi

if [ $1 = "stream" ]; then
  echo "Processing FashionMNIST in the stream mode"

  # streaming version
  python streamIDT.py FashionMNIST  --memory_usage 2 --root_data_dir qsketch/data/FashionMNIST  --clip $CLIPTO --num_quantiles $NUM_QUANTILES  --projectors $PROJECTORS  --input_dim $INPUT_DIM --num_samples 3000 --plot --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS --batchsize $BATCHSIZE --plot_dir samples_fmnist_stream_$PROJECTORS --log --logdir logs/fmnist/stream/$PROJECTORS
fi
