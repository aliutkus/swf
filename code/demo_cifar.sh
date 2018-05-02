python qsketch/sketch.py CIFAR10  -n 40 -q 100 -o ~/40sketchCIFAR10 -s 32 -p RandomProjectors -m 1 -r qsketch/data/CIFAR10

python sketchIDT.py ~/40sketchCIFAR10.npy  -d 300 -n 3000 --plot -s 1 -r 1e-4 -e 10000 -b 10
