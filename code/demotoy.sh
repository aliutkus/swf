# generate the toy data
python generate_toydata.py  --output toy --dim 2 --num_samples 50000 --num_components 10 --seed 0

# launch SWF for the density plots
python swf.py toy --num_sketches 1 --num_examples 5000 --num_quantiles 100 --input_dim 2  --num_samples 5000 --stepsize 1 --regularization 0 --num_thetas 30 --num_epochs 50 --plot_dir ~/swf_samples_toy --plot_epochs 2 3 5 10 20 50  --no_particles_plot --no_swcost_plot
