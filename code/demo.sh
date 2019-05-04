# parameters for sketching
NUM_SKETCHES=10
NUM_THETAS=4000
NUM_QUANTILES=500
NUM_EXAMPLES=5000


# images dimensions
IMG_SIZE=64

# parameters for auto encoder
AE_STRING="--ae"
CONV_AE_STRING="--conv_ae"
BOTTLENECK_SIZE=64

# parameters for SWF
# pick something like num_thetas/4
STEPSIZE=5
REG=0
NUM_EPOCHS=5000

# whether to change the sketches at each epoch or not.
# to change the sketch, set this to "--no_fixed_sketch"
NO_FIXED_SKETCH_STRING=""

# number of particles
NUM_SAMPLES=5000

# controls the dimension of the input. -1 for the dimension of the bottleneck
INPUT_DIM=10

# number of test
NUM_TEST=0
TEST_TYPE='RANDOM'

# plot options
PLOT_EVERY=10
MATCH_EVERY=500

if [ $1 = "toy" ]; then
  echo "generating toy data, and then SWF on it"

  # Generate data:
  NUM_SAMPLES_TOY=50000
  NUM_COMPONENTS_TOY=10
  INPUT_DIM=2
  SEED_STRING="--seed 0"

  python ./generate_toydata.py  --output toy --dim $INPUT_DIM --num_samples $NUM_SAMPLES_TOY --num_components $NUM_COMPONENTS_TOY $SEED_STRING
fi

# now launch the sliced Wasserstein flow
python -W ignore swf.py $1 $NO_FIXED_SKETCH_STRING --num_workers 15 --root_data_dir ~/data --img_size $IMG_SIZE --num_sketches $NUM_SKETCHES --num_examples $NUM_EXAMPLES --num_quantiles $NUM_QUANTILES --input_dim $INPUT_DIM  --num_samples $NUM_SAMPLES --stepsize $STEPSIZE --regularization $REG --num_thetas $NUM_THETAS --plot_dir ~/swf_samples_$1 --plot_every $PLOT_EVERY --num_epochs $NUM_EPOCHS --match_every $MATCH_EVERY $AE_STRING $CONV_AE_STRING --bottleneck_size $BOTTLENECK_SIZE --ae_model ae --num_test $NUM_TEST --test_type $TEST_TYPE
