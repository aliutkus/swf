# python qsketch/sketch.py CIFAR10  -n 400 -q 100 -o ~/tmp/sketchCIFAR10 -s 32 -p RandomProjectors -m 1 -r qsketch/data/CIFAR10
#
#python sketchIDT.py ~/sketchCIFAR10.npy  -d 300 -n 3000 --plot -s 100 -r 0.01 -e 10000 -b 5

#python qsketch/sketch.py CIFAR10  -n 1000 -q 100 -o ~/tmp/localizedCIFAR10 -s 32 -p RandomLocalizedProjectors -m 1 -r qsketch/data/CIFAR10
python sketchIDT.py ~/tmp/localizedCIFAR10.npy  -d 1000 -n 3000 --plot -s 1000 -r 0.001 -e 10000 -b 10 -p samples_localized
