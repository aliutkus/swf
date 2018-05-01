# Generate data:
 python ./generate_toydata.py  -o toydata -d 20 -n 50000 -c 20 -s 20

# Sketch
 python qsketch/sketch.py ./toydata.npy -n 400 -q 100 -o toysketch -p RandomProjectors -x 10

# IDT
 python sketchIDT.py toysketch.npy  -d 20 -n 1000 --plot_target toydata.npy --plot -s 1 -r 2 -e 100
