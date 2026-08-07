"""Matplotlib examples for data visualization."""

import matplotlib.pyplot as plt
import numpy as np


def example_simple_plot():
    xpoints = np.array([0, 6])
    ypoints = np.array([0, 250])
    plt.plot(xpoints, ypoints)
    plt.title("Simple Line Plot")
    plt.show()


def example_multiple_plot():
    xpoint = np.array([1, 4, 5, 7])
    ypoint = np.array([3, 9, 6, 14])
    plt.plot(xpoint, ypoint)
    plt.title("Multiple Plot Points")
    plt.show()


def example_marker_plot():
    xpoint = np.array([1, 4, 5, 7])
    ypoint = np.array([3, 9, 6, 14])
    plt.plot(xpoint, ypoint, 'd', c='hotpink')
    plt.title("Marker Plot")
    plt.show()


def example_styled_plot():
    xpoint = np.array([1, 4, 5, 7])
    ypoint = np.array([3, 9, 6, 14])
    plt.plot(xpoint, ypoint, ls='dashdot', c='hotpink', lw=11)
    plt.grid()
    plt.title("Styled Line Plot")
    plt.show()


def example_labeled_plot():
    xpoint = np.array([1, 4, 5, 7])
    ypoint = np.array([3, 9, 6, 14])
    plt.plot(xpoint, ypoint, ls='dashdot', c='hotpink', lw=1)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(True)
    plt.title("Labeled Plot")
    plt.show()


if __name__ == "__main__":
    example_simple_plot()
    example_multiple_plot()
    example_marker_plot()
    example_styled_plot()
    example_labeled_plot()
