# script to test the SWF with toy data (python 3.6).
# dependencies:
#joblib
#seaborn
#numpy
#scipy
#matplotlib
#seaborn
#-------------------------------------


# parameters for data generation
DIMENSION=5
NUM_COMPONENTS=20

# parameters for computation of the data CDF
NUM_QUANTILES=100
NUM_THETAS=30
CLIPTO=5000

# parameters for SWF
STEPSIZE=5
REG=0.001

# input dimension. If different from the GMM dimension, will multiply by
# a random matrix.
INPUT_DIM=5

# number of iterations
STOP=500

# uncomment to save the plots in the "output" directory
PLOTDIR_STRING=""
#PLOTDIR_STRING="output"

# uncomment to get the example for the article (needs to be DIMENSION=5)
#SEED_STRING="--seed 0"

# will display a contour plot regularly
CONTOUR_EVERY=50

# Generate data:
 python ./generate_toydata.py  --output toydata --dim $DIMENSION --num_samples 50000 --num_components $NUM_COMPONENTS $SEED_STRING

# apply SWF
python SWF.py ./toydata.npy --stop $STOP $PLOTDIR_STRING --contour_every $CONTOUR_EVERY --clip $CLIPTO --num_quantiles $NUM_QUANTILES  --input_dim $INPUT_DIM  --num_samples 3000 --stepsize $STEPSIZE --reg $REG --num_thetas $NUM_THETAS
