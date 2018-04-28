 python ./generate_toydata.py  -o toydata -d 2 -n 50000 -c 30
 python qsketch/sketch.py ./toydata.npy -n 30 -q 300 -o toysketch
 python sketchIDT.py toysketch.npy  -d 2 -n 3090 --plot_target toydata.npy --plot -r 2
