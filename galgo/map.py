import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.patch.set_alpha(0)
ax2 = ax1.twinx()
im = matplotlib.image.imread('map.png')
ax2.imshow(im)
ax1.set_zorder(1) #grid is shown but image disappears

plt.show()
