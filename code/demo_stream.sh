# Generate data:
 python ./generate_toydata.py  --output toydata --dim 10 --num_samples 50000 --num_components 20 --seed 0

# Sketch
 python streamIDT.py ./toydata.npy  --num_quantiles 200  --projectors RandomLocalizedProjectors  --input_dim 20 --num_samples 3000 --plot_target toydata.npy --plot --stepsize 1 --reg 0.001 --batchsize 5
