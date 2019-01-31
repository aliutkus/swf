# parameters for sketching
NUM_SKETCHES=10
NUM_QUANTILES=100
# pick something related to bottleneck size for num_thetas
NUM_THETAS=1000
CLIPTO=3000

IMG_SIZE=32
# parameters for auto encoder
BOTTLENECK_SIZE=32
AE_STRING="--ae"
CONV_AE_STRING="--conv_ae"

# parameters for SWF
# pick something like num_thetas/4
STEPSIZE=50
REG=0
NUM_EPOCHS=1001

# parameters for particles
NUM_SAMPLES=3000
# particles type is either TESTSET or RANDOM. If RANDOM, the RANDOM_INPUT_DIM
# controls the dimension of the input
PARTICLES_TYPE='RANDOM'
INPUT_DIM=-1

# number of test
NUM_TEST=0
TEST_TYPE='INTERPOLATE'

# plot every
PLOT_EVERY=100
MATCH_EVERY=1000

if [ $1 = "toy.npy" ]; then
  echo "generating toy data, and then SWF on it"

  # Generate data:
  NUM_SAMPLES_TOY=50000
  NUM_COMPONENTS_TOY=5
  SEED_STRING="--seed 10"

  python ./generate_toydata.py  --output toy --dim $INPUT_DIM --num_samples $NUM_SAMPLES_TOY --num_components $NUM_COMPONENTS_TOY $SEED_STRING
fi

# now launch the sliced Wasserstein flow
python -m cProfile -o 100_percent_gpu_utilization.prof swf.py $1 --root_data_dir ~/data/$1 --img_size $IMG_SIZE --num_sketches $NUM_SKETCHES --clip $CLIPTO --num_quantiles $NUM_QUANTILES --input_dim $INPUT_DIM  --num_samples $NUM_SAMPLES --particles_type=$PARTICLES_TYPE --stepsize $STEPSIZE --num_thetas $NUM_THETAS --plot_dir ~/swf_samples_$1 --logdir logs/$1 --plot_every $PLOT_EVERY --num_epochs $NUM_EPOCHS --match_every $MATCH_EVERY $AE_STRING $CONV_AE_STRING --bottleneck_size $BOTTLENECK_SIZE --ae_model ae --num_test $NUM_TEST --test_type $TEST_TYPE
