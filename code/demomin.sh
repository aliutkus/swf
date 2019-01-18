# parameters for sketching
NUM_QUANTILES=200
NUM_THETAS=500
CLIPTO=5000

# parameters for SWF
INPUT_DIM=32
BOTTLENECK_SIZE=32
NUM_SAMPLES=5000
NUM_SKETCHES=200
STEPSIZE=0.01
AE_STRING="--ae"

if [ $1 = "toy.npy" ]; then
  echo "generating toy data, and then SWF on it"

  # Generate data:
  NUM_SAMPLES_TOY=50000
  NUM_COMPONENTS_TOY=5
  SEED_STRING="--seed 10"

  python ./generate_toydata.py  --output toy --dim $INPUT_DIM --num_samples $NUM_SAMPLES_TOY --num_components $NUM_COMPONENTS_TOY $SEED_STRING
fi

# now launch the sliced Wasserstein flow
python swf.py $1 --root_data_dir ~/data/$1 --img_size 32 --num_sketches $NUM_SKETCHES --clip $CLIPTO --num_quantiles $NUM_QUANTILES --input_dim $INPUT_DIM  --num_samples $NUM_SAMPLES --stepsize $STEPSIZE --num_thetas $NUM_THETAS --plot_dir ~/swmin_samples_$1 --logdir logs/$1 --plot_every 50 $AE_STRING --bottleneck_size $BOTTLENECK_SIZE --ae_model ae --swmin
