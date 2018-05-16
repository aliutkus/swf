import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig = plt.figure(num=1, figsize=(5, 3))
sns.set_style('whitegrid')

d = np.load('logs.npy').item()
ws = np.array(d['errors'])
ws2 = ws[[0,1,2,3,5,6],:]

limy = [ws2.min(), ws2.max()]
limx = [0, 500]
plt.xlim(limx)
plt.ylim(limy)
ax = plt.gca()
ticks = [np.linspace(*v, 5) for v in [limx, limy]]

ax.set(xticks=ticks[0], yticks=ticks[1])


plt.plot(ws2.T)
ax.set_yscale("log", nonposy='clip')
plt.legend(['d=2', 'd=5', 'd=10', 'd=50', 'd=100', 'd=200'], ncol=2, frameon=True)
plt.xlabel('iteration $k$')
plt.ylabel('sliced Wasserstein distance')
plt.tight_layout()
plt.show()
