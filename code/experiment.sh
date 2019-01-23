parallel --bar -j 1 python swf.py {3} --root_data_dir "/home/antoine/data/{3}" --img_size 32 --num_sketches 1 --clip 50000 --num_quantiles 100 --input_dim {2} --num_samples 3000 --particles_type=RANDOM --stepsize 500 --num_thetas 16000 --plot_dir "/home/antoine/swf_samples_{3}_bottleneck{1}_inputdim{2}"  --plot_every 100 --match_every -1 --ae --conv_ae --bottleneck_size {1} --ae_model ae --num_test 0 --test_type RANDOM --num_iter 1001 \
::: 16 32 48 64 80 \
::: -1 \
::: MNIST
