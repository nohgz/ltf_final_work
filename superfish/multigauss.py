import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from numba import njit

def fit_gaussian_density(
    z_array: NDArray[np.float64],
    nbins: int = 50,
    ngaussians: int = 50,
    width: float = 0.00005,
    bunch_charge: float = None,
    plot: bool = False):
    """
    Fit a sum of Gaussians to a 1D array (i.e. particle z-positions),
    return (xGauss, ampGauss, sigGauss) for use in space charge solver.
    """

    if bunch_charge is None:
        raise ValueError("You must provide a bunch charge!")

    # Histogram particle positions
    histo, bins = np.histogram(z_array, bins=nbins, density=True)

    bin_width = bins[1] - bins[0]
    guesses = np.linspace(bins[1], bins[-2], ngaussians)

    xGauss = np.zeros(ngaussians, dtype=np.float64)
    ampGauss = np.zeros(ngaussians, dtype=np.float64)
    sigGauss = np.full(ngaussians, width, dtype=np.float64)

    normsum = 0.0

    for i in range(ngaussians):
        idx = int((guesses[i] - bins[0]) / bin_width)
        idx = np.clip(idx, 0, len(histo) - 1)

        xGauss[i] = (bins[idx] + bins[idx+1]) / 2
        ampGauss[i] = histo[idx]
        normsum += ampGauss[i] * (width * np.sqrt(2 * np.pi))

    ampGauss /= normsum

    if plot:
        mesh = np.linspace(bins[0], bins[-1], 1000)
        histo *= bunch_charge
        fitted_line = np.zeros_like(mesh)

        for i in range(ngaussians):
            fitted_line += ampGauss[i] * np.exp(-(mesh - xGauss[i]) ** 2 / (2 * width**2))
        fitted_line *= bunch_charge

        # calculate a test value using the things
        plt.figure(figsize=(7, 4))
        plt.stairs(histo, bins, label="Histogram")
        plt.plot(mesh, fitted_line, 'r-', label="Gaussian Fit")
        plt.title("Fitted $\\rho$ [$C$]")
        plt.xlabel("z [$m$]")
        plt.ylabel("Charge Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        total_charge = np.trapz(fitted_line, mesh)
        print(f"Integral = {total_charge} C, Provided Bunch Charge = {bunch_charge}")

    return xGauss, ampGauss, sigGauss

@njit(fastmath=True)
def _gauss_sum(x,NGauss,xGauss,ampGauss,sigGauss):
    NGauss = len(xGauss)
    Sum = 0
    normsum = 0.0
    for i in range(NGauss):
        Sum += ampGauss[i]*np.exp(-(x-xGauss[i])**2/2/sigGauss[i]**2)
        normsum += ampGauss[i]*np.sqrt(2*np.pi*sigGauss[i]**2)
    return Sum/normsum

@njit(fastmath=True)
def _lmd(z,xGauss,ampGauss,sigGauss):
    NGauss = len(xGauss)
    return _gauss_sum(z,NGauss,xGauss,ampGauss,sigGauss)