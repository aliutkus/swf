# Generate data:
 python ./generate_toydata.py  -o toydata -d 3 -n 50000 -c 20 --seed 0

# Sketch
 python qsketch/sketch.py ./toydata.npy -n 500 -q 100 -o toysketch -p RandomProjectors

# IDT
 python sketchIDT.py toysketch.npy  -d 3 -n 10000 --plot_target toydata.npy --plot -s 1 -r 1 -e 10000 -b 10
