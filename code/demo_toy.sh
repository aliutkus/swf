# script to test the SWF with toy data.
# Usage: `demo_toy.sh stream` for the streaming version
#        `demo_toy.sh batch` for the batch version (fixed number of sketches)
#        `demo_toy.sh batch save` for saving the computing chain for later use
#        `demo_toy.sh batch load` for using a computing chain.

# parameters for data generation
DIMENSION=2
NUM_COMPONENTS=20
NUM_SAMPLES=50000

# parameters for sketching
NUM_QUANTILES=50
NUM_THETAS=50
CLIPTO=1000

# parameters for SWF
NUM_PARTICLES=3000
STEPSIZE=1
REG=0.01
INPUT_DIM=2
BATCHSIZE=1

EPOCHS=1

# if we want to save the plots, uncomment the following line
#PLOTDIR_STRING="--plot_dir toy_P"$NUM_SAMPLES"_N"$NUM_PARTICLES"_Q"$NUM_QUANTILES"_CLIPTO"$CLIPTO"_REG_"$REG
#else:
PLOTDIR_STRING=""

PLOT_STRING="--plot"
SEED_STRING="--seed 10"
CONTOUR_EVERY=100
STOP=5000

# Generate data:
 python ./generate_toydata.py  --output toydata --dim $DIMENSION --num_samples $NUM_SAMPLES --num_components $NUM_COMPONENTS $SEED_STRING

if [ $1 = "batch" ]; then
  # batch version
  echo "Processing toy data in the batch mode"

  if [ "$PLOTDIR_STRING" != "" ]; then
    PLOTDIR_STRING=$PLOTDIR_STRING"_batch"
  fi

  NUM_SKETCHES=100
   if [ $2 = "save" ]; then
     echo " ... will save the chain."
     CHAIN_STRING="--output_chain chain_save"
     if [ "$PLOTDIR_STRING" != "" ]; then
       PLOTDIR_STRING=$PLOTDIR_STRING"_save"
     fi
   else
     CHAIN_STRING=""
   fi
   if [ $2 != "load" ]; then
     # Sketch
      python qsketch/sketch.py ./toydata.npy --num_sketches $NUM_SKETCHES --clip $CLIPTO --num_thetas $NUM_THETAS --num_quantiles $NUM_QUANTILES --output toysketch --projectors RandomProjectors

     # SWF
     python batchIDT.py toysketch.npy $PLOT_STRING --contour_every $CONTOUR_EVERY --input_dim $INPUT_DIM --num_samples $NUM_PARTICLES --plot_target toydata.npy  --stepsize $STEPSIZE --batchsize $BATCHSIZE --reg $REG --epochs $EPOCHS --log  --logdir logs/toy/batch/$NUMTHETAS $CHAIN_STRING $PLOTDIR_STRING
   else
     echo " ... using previously saved chain."
     if [ "$PLOTDIR_STRING" != "" ]; then
       PLOTDIR_STRING=$PLOTDIR_STRING"_load"
     fi
     # use a previously computed chain
     python batchIDT.py toysketch.npy  $PLOT_STRING --input_chain chain_save.npy --input_dim $INPUT_DIM --num_samples 3000 --plot_target toydata.npy $PLOTDIR_STRING
   fi
fi

if [ $1 = "stream" ]; then
  # streaming version
  echo "Processing toy data in the stream mode"
  if [ "$PLOTDIR_STRING" != "" ]; then
    PLOTDIR_STRING=$PLOTDIR_STRING"_stream"
  fi
  python streamIDT.py ./toydata.npy  --stop $STOP $PLOT_STRING --clip $CLIPTO --num_quantiles $NUM_QUANTILES  --projectors RandomProjectors  --input_dim $INPUT_DIM --batchsize $BATCHSIZE --num_samples $NUM_PARTICLES --plot_target toydata.npy  --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS  --log --logdir logs/toy/stream $PLOTDIR_STRING
fi
