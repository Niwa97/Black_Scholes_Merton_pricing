import numpy as np
from numpy.random import default_rng
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt

S0 = 35.0
K = 40.0
r = 0.03
sigma = 0.1
T_max = 2.0

number_of_sim = 100000
max_steps = 100

dt = T_max/max_steps
dr = np.exp(-r*dt)

rng = np.random.default_rng(10)
brownian_motion = rng.standard_normal((max_steps + 1, number_of_sim))

action_prices = np.zeros_like(brownian_motion)
action_prices[0] = S0

for n in range (1, max_steps+1):
  action_prices[n] = action_prices[n-1] * np.exp( (r - (sigma**2)/2.0) * dt + sigma * np.sqrt(dt) * brownian_motion[n] )

plt.plot(action_prices[:,:20])
plt.show()

final_prices = action_prices[-1]
plt.hist(final_prices, bins = 40)
plt.axvline(final_prices.mean(), label = 'mean', color = 'y')
plt.axvline(final_prices.mean() - final_prices.std(), label = 'mean - std_dev', color = 'r')
plt.axvline(final_prices.mean() + final_prices.std(), label = 'mean + std_dev', color = 'g')
plt.legend()
plt.show()

H_european_put = np.maximum(K - final_prices, 0)
plt.hist(H_european_put, bins = 40)
plt.show()
payoff_european = dr*H_european_put.mean()
print("European put: ", payoff_european)

H_american_put = np.maximum(K - action_prices, 0)
values = H_american_put[-1]
for n in range(max_steps-1, 0, -1):
  least_squares = Polynomial.fit(action_prices[n], dr*values, deg = 4)
  fitted_value = least_squares(action_prices[n])
  values = np.where(H_american_put[n] > fitted_value, H_american_put[n], dr*values)

payoff_american = dr*H_american_put.mean()
print("American put: ", payoff_american)
