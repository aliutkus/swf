# Generate data:
 python ./generate_toydata.py  --output toydata --dim 5 --num_samples 50000 --num_components 20 --seed 0

# Sketch
 python streamIDT.py ./toydata.npy  --clip 5000 --num_quantiles 100  --projectors RandomProjectors  --input_dim 20 --num_samples 3000 --plot_target toydata.npy  --stepsize 0.5 --reg 0.001 --num_thetas 200 --logdir logs/stream
