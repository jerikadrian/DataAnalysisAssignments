import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math

def compute_slope_and_se(x, y):
    res = linregress(x, y)
    return res.slope, res.intercept, res.rvalue, res.stderr
    
# MAIN ANALYSIS:

# Interaction Hypothesis — Does humidity change how temperature affects daily energy consumption in the household?
# This is more insightful than a simple t-test because we are asking:
# “Is the effect of temperature on energy DIFFERENT depending on humidity?”

# FORMAL HYPOTHESIS:
# H0:  β_low  =  β_high
#      (the slope of Temp→Energy is the same on low- and high-humidity days)
# H1:  β_low  ≠  β_high
#      (the temperature–energy relationship changes with humidity level)

# We test this by:
#  1) Splitting the data into low- and high-humidity groups
#  2) Fitting a regression line for each group
#  3) Testing the difference in slopes using a Z-test
#  4) Creating two figures:
#       Figure 1 — scatter plots + regression lines for both groups
#       Figure 2 — bar chart comparing slopes with error bars

def humidity_interaction_analysis(df):

    median_h = df["h_out"].median()
    
    low_h  = df[df["h_out"] <= median_h]
    high_h = df[df["h_out"] >  median_h]

    x_low  = low_h["t_in"]
    x_high = high_h["t_in"]

    y_low  = low_h["total_energy"]
    y_high = high_h["total_energy"]

    slope_low,  a_low,  r_low,  se_low  = compute_slope_and_se(x_low,  y_low)
    slope_high, a_high, r_high, se_high = compute_slope_and_se(x_high, y_high)

    Z = (slope_low - slope_high) / np.sqrt(se_low**2 + se_high**2)

    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(Z) / math.sqrt(2))))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    axs[0].scatter(x_low, y_low, alpha=0.5, color='blue')
    axs[0].plot(x_low, a_low + slope_low * x_low, color='black', lw=2)
    axs[0].set_title("Low Humidity Days")
    axs[0].set_xlabel("Indoor Temperature (°C)")
    axs[0].set_ylabel("Daily Energy Use")

    axs[0].text(0.05, 0.85,
                f"Slope = {slope_low:.4f}\nr = {r_low:.3f}",
                transform=axs[0].transAxes,
                bbox=dict(color='white', alpha=0.8))

    axs[1].scatter(x_high, y_high, alpha=0.5, color='red')
    axs[1].plot(x_high, a_high + slope_high * x_high, color='black', lw=2)
    axs[1].set_title("High Humidity Days")
    axs[1].set_xlabel("Indoor Temperature (°C)")

    axs[1].text(0.05, 0.85,
                f"Slope = {slope_high:.4f}\nr = {r_high:.3f}",
                transform=axs[1].transAxes,
                bbox=dict(color='white', alpha=0.8))

    fig.suptitle("Figure 2. Temperature–Energy Relationship Under Low vs High Humidity",
                 fontsize=12)

    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 5), tight_layout=True)

    ax2.bar(["Low Humidity", "High Humidity"],
            [slope_low, slope_high],
            yerr=[se_low, se_high],
            color=["blue", "red"],
            alpha=0.7,
            capsize=8)

    ax2.set_ylabel("Regression Slope")
    ax2.set_title("Figure 3. Comparison of Slopes (Temperature → Energy)")

    # Add the p-value below the plot
    ax2.text(0.5, -0.12,
             f"p-value for slope difference = {p_value:.4f}",
             ha='center', transform=ax2.transAxes, fontsize=10)

    plt.show()

    print("---- REGRESSION RESULTS ----")
    print(f"Low-humidity slope:  {slope_low:.5f}  (SE = {se_low:.5f})")
    print(f"High-humidity slope: {slope_high:.5f} (SE = {se_high:.5f})")
    print()
    print(f"Z statistic: {Z:.4f}")
    print(f"p-value:     {p_value:.4f}")
    print("----------------------------")
