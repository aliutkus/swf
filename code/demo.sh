 python ./generate_toydata.py  -o toydata -d 2 -n 50000 -c 20 -s 20
 python qsketch/sketch.py ./toydata.npy -n 400 -q 100 -o toysketch -p UnitaryProjectors -x 1
 python sketchIDT.py toysketch.npy  -d 2 -n 10000 --plot_target toydata.npy --plot -r 1 -e 100
