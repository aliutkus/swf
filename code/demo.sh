# Generate data:
 python ./generate_toydata.py  -o toydata -d 5 -n 50000 -c 20 --seed 0

# Sketch
 python qsketch/sketch.py ./toydata.npy -n 50 -q 100 -o toysketch -p RandomLocalizedProjectors

# IDT
 python sketchIDT.py toysketch.npy  -d 2 -n 3000 --plot_target toydata.npy --plot -s 1 -r 0.001 -e 10000 -b 10
