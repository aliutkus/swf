# Generate data:
 python ./generate_toydata.py  --output toydata --dim 10 --num_samples 50000 --num_components 20 --seed 0

# Sketch
 python qsketch/sketch.py ./toydata.npy --num_sketches 50 --clip 5000 --num_thetas 200 --num_quantiles 100 --output toysketch --projectors RandomProjectors

# IDT
 python batchIDT.py toysketch.npy  --input_dim 2 --num_samples 3000 --plot_target toydata.npy --plot --stepsize 1 --reg 0.001 --epochs 100
