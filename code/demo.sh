# Generate data:
 python ./generate_toydata.py  -o toydata -d 6 -n 50000 -c 20 -s 20

# This doesn't work sketch with more projections than dimensions
# python qsketch/sketch.py ./toydata.npy -n 400 -q 100 -o toysketch -p RandomProjectors -x 2

# This work:
 python qsketch/sketch.py ./toydata.npy -n 400 -q 100 -o toysketch -p RandomProjectors -x 1

# IDT
 python sketchIDT_umut.py toysketch.npy  -d 6 -n 10000 --plot_target toydata.npy --plot -r 1 -e 100
