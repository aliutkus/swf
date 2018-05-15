# batch version
#python qsketch/sketch.py FashionMNIST --num_sketches 500 --num_quantiles 100 --clip 5000 --num_thetas 500 --output ~/tmp/sketchFashionMNIST --projectors RandomProjectors --memory_usage 2 --root_data_dir qsketch/data/FashionMNIST
#python batchIDT.py ~/tmp/sketchFashionMNIST.npy  --input_dim 2000 --num_samples 3000  --plot --stepsize 300 --reg 0.0001 --epochs 100 --plot_dir samples_fmnist_batch


# streaming version
python streamIDT.py FashionMNIST  --memory_usage 2 --root_data_dir qsketch/data/FashionMNIST  --clip 3000 --num_quantiles 100  --projectors RandomProjectors  --input_dim 2000 --num_samples 3000 --plot_target toydata.npy --plot --stepsize 300 --reg 0.0001 --num_thetas 400 --plot_dir samples_fmnist_stream
