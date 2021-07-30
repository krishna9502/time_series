import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(left=-2, right=100)

## random walk with gauusian noise
## y_t = y_t_1 + z_t
## y_0_1 = 0

# y_t_1 = 0

# z = []
# y = []
# line, = ax.plot(y)

# for i in range(100):
#     ax.lines.remove(line)
    
#     z_t = np.random.normal(0,1.0)
#     y_t = y_t_1 + z_t
#     z.append(z_t)
#     y.append(y_t)

#     y_t_1 = y_t

#     line, = ax.plot(np.arange(i+1), y, color='blue')
#     plt.draw()
#     plt.pause(1)
# plt.pause(0)

## random walk with unifrom noise

y_t_1 = 0

z = []
y = []
line, = ax.plot(y)

for i in range(100):
    ax.lines.remove(line)
    
    z_t = np.random.uniform(-1.0,1.0)
    y_t = y_t_1 + z_t
    z.append(z_t)
    y.append(y_t)

    y_t_1 = y_t

    line, = ax.plot(np.arange(i+1), y, color='blue')
    plt.draw()
    plt.pause(1)
plt.pause(0)
