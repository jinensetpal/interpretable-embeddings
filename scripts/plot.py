#!/usr/bin/env python3

import matplotlib.pyplot as plt

x = [12, 48, 96, 384, 512, 768]
plt.plot(x, [.99056, .99046, .98947, .98997, .99076, .98997], label='AutoEncoder (online)')
plt.plot(x, [.77993, .69250, .63130, .87690, .88731, .94834], label='PCA')
plt.plot(x, [.93674, .93987, .94205, .95689, .95842, .95765], label='UMAP')

plt.legend()
plt.savefig('scores.png')
