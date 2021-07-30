import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(left=-2, right=500)

## AR process with gaussian noise
## y_t = y_t-1 + y_t-2 + nu_t
## nu ~ gaussian(mu=0,var=1)
## y(-1) = 0, y(-2) = 0

# y_t_1 = 0
# y_t_2 = 0

# nu = []
# y = []
# line, = ax.plot(y)

# a1 = 0.1
# a2 = 0.1

# for i in range(100):
#     ax.lines.remove(line)
    
#     nu_t = np.random.normal(0,0.1)
#     y_t = a1 * y_t_1 + a2 * y_t_2 + nu_t
#     nu.append(nu_t)
#     y.append(y_t)

#     y_t_2 = y_t_1
#     y_t_1 = y_t

#     line, = ax.plot(np.arange(i+1), y, color='blue')
#     plt.draw()
#     plt.pause(0.1)
# plt.pause(0)

## MA process with gaussian noise
## y_t = nu_t + a1 * nu_t_1 + a2 * nu_t_2
## nu ~ gaussian(mu=0,var=1)

# nu_t_1 = np.random.normal(0,0.1)
# nu_t_2 = np.random.normal(0,0.1)

# nu = []
# y = []
# line, = ax.plot(y)

# a1 = 1
# a2 = -1

# for i in range(100):
#     ax.lines.remove(line)
    
#     nu_t = np.random.normal(0,0.1)
#     y_t = nu_t + a1 * nu_t_1 + a2 * nu_t_2
#     nu.append(nu_t)
#     y.append(y_t)

#     nu_t_2 = nu_t_1
#     nu_t_1 = nu_t

#     line, = ax.plot(np.arange(i+1), y, color='blue')
#     plt.draw()
#     plt.pause(0.1)
# plt.pause(0)

## ARMA process with gaussian noise
## y_t + a1 * y_t_1 + a2 * y_t_2 = nu_t + b1 * nu_t_1 + b2 * nu_t_2
## nu ~ gaussian(mu=0,var=1)

y_t_1 = 0
y_t_2 = 0

nu_t_1 = np.random.normal(0,0.1)
nu_t_2 = np.random.normal(0,0.1)

nu = []
y = []
line, = ax.plot(y)

a1, a2, b1, b2 = (0.1, 0.1, 0.1, 0.1)

for i in range(500):
    ax.lines.remove(line)
    
    nu_t = np.random.normal(0,0.1)
    y_t = a1 * y_t_1 + a2 * y_t_2 + nu_t + b1 * nu_t_1 + b2 * nu_t_2
    nu.append(nu_t)
    y.append(y_t)

    y_t_2 = y_t_1
    y_t_1 = y_t
    nu_t_2 = nu_t_1
    nu_t_1 = nu_t

    line, = ax.plot(np.arange(i+1), y, color='blue')
    plt.draw()
    plt.pause(0.1)
plt.pause(0)

