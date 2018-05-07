#python qsketch/sketch.py FashionMNIST  -n 500 -q 100 -o ~/tmp/localizedsketchFashionMNIST -s 32 -p RandomLocalizedProjectors -m 1 -r qsketch/data/FashionMNIST

python sketchIDT.py ~/tmp/localizedsketchFashionMNIST.npy  -d 500 -n 3000 --plot -s 100 -r 0.0001 -e 10000 -b 5 -p samples_fmnist_localized
python sketchIDT.py ~/tmp/sketchFashionMNIST.npy  -d 500 -n 3000 --plot -s 100 -r 0.0001 -e 10000 -b 5 -p samples_fmnist
