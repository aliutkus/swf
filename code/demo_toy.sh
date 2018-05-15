# parameters for data generation
DIMENSION=5
NUM_COMPONENTS=20

# parameters for sketching
NUM_QUANTILES=100
NUM_THETAS=100
CLIPTO=5000

# parameters for SWF
STEPSIZE=1
REG=0.001
INPUT_DIM=20
BATCHSIZE=5

# Generate data:
 python ./generate_toydata.py  --output toydata --dim $DIMENSION --num_samples 50000 --num_components $NUM_COMPONENTS --seed 0

if [ $1 = "batch" ]; then
  # batch version
  echo "Processing toy data in the batch mode"

  NUM_SKETCHES=500

  # Sketch
   python qsketch/sketch.py ./toydata.npy --num_sketches $NUM_SKETCHES --clip $CLIPTO --num_thetas $NUM_THETAS --num_quantiles $NUM_QUANTILES --output toysketch --projectors RandomProjectors

  # IDT
   python batchIDT.py toysketch.npy  --input_dim $INPUT_DIM --num_samples 3000 --plot_target toydata.npy  --stepsize $STEPSIZE --batchsize $BATCHSIZE --reg $REG --epochs 100 --logdir logs/toy/batch
fi

if [ $1 = "stream" ]; then
  # streaming version
  echo "Processing toy data in the stream mode"

  python streamIDT.py ./toydata.npy  --clip $CLIPTO --num_quantiles $NUM_QUANTILES  --projectors RandomProjectors  --input_dim $INPUT_DIM --batchsize $BATCHSIZE --num_samples 3000 --plot_target toydata.npy  --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS --logdir logs/toy/stream
fi
