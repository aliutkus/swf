python qsketch/sketch.py CIFAR10  -n 400 -q 100 -o ~/sketchCIFAR10 -s 32 -p RandomProjectors -m 1 -r qsketch/data/CIFAR10

python sketchIDT.py ~/sketchCIFAR10.npy  -d 30 -n 3000 --plot -s 1 -r 1 -e 10000 -b 10
