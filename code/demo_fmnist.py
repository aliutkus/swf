### batch version
#python qsketch/sketch.py FashionMNIST --num_sketches 5 --num_quantiles 100 --output ~/tmp/localizedsketchFashionMNIST --projectors RandomLocalizedProjectors --memory_usage 1 --root_data_dir qsketch/data/FashionMNIST
#python batchIDT.py ~/tmp/localizedsketchFashionMNIST.npy  --input_dim 500 --num_samples 3000  --plot --stepsize 100 --reg 0.0001 --epochs 100 --batchsize 5 --plot_dir samples_fmnist_localized


# streaming version
python streamIDT.py FashionMNIST  --memory_usage 1 --root_data_dir qsketch/data/FashionMNIST --num_quantiles 200  --projectors RandomLocalizedProjectors  --input_dim 500 --num_samples 3000  --plot --stepsize 100 --reg 0.0001 --batchsize 5 --plot_dir samples_fmnist_localized
