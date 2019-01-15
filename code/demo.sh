# parameters for sketching
NUM_QUANTILES=1000
NUM_THETAS=5000
CLIPTO=500

# parameters for SWF
INPUT_DIM=5
NUM_SAMPLES=500

STEPSIZE=100
REG=0.0001

if [ $1 = "toy.npy" ]; then
  echo "generating toy data, and then SWF on it"

  # Generate data:
  NUM_SAMPLES_TOY=50000
  NUM_COMPONENTS_TOY=5
  SEED_STRING="--seed 10"

  python ./generate_toydata.py  --output toy --dim $INPUT_DIM --num_samples $NUM_SAMPLES_TOY --num_components $NUM_COMPONENTS_TOY $SEED_STRING
fi

# now launch the sliced Wasserstein flow
python swf.py $1  --root_data_dir ~/data/CIFAR10  --clip $CLIPTO --num_quantiles $NUM_QUANTILES   --input_dim $INPUT_DIM --num_samples $NUM_SAMPLES --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS --plot_dir ~/samples_$1 --logdir logs/$1
