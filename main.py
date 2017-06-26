'''
* Created by Zhenia Syryanyy (Yevgen Syryanyy)
* e-mail: yuginboy@gmail.com
* License: this code is in GPL license
* Last modified: 2017-06-23
'''

import numpy as np
import pickle  # for loading pickled test data
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import warnings
from libs.backgrounds import bg_subtraction_recursively, bg_move_curve_to_zero_line

from scipy.optimize import differential_evolution

ln_2 = math.log(2)


def f_GL(x, F, E, m):
    # Gaussian/Lorentzian Product Form
    #  Gaussian/Lorentzian product formula where the mixing is determined by m = p/100, GL(100) is a pure Lorentzian
    #  while GL(0) is pure Gaussian.
    return np.exp(-4 * ln_2 * (1 - m) * ((x - E) ** 2) / (F ** 2)) / (1 + 4 * m * ((x - E) ** 2) / (F ** 2))


def f_composition(x, F, E, m):
    # composition (Sum of functions)
    N = np.size(F)
    sum = 0
    for i in range(N):
        sum = sum + f_GL(x, F[i], E[i], m[i])

    return sum


def std(x, y, F, E, m):
    return np.sqrt(np.sum((y - f_composition(x, F, E, m)) ** 2)) / np.size(x)


# Double Lorentzian peak function
# bounds on parameters are set in generate_Initial_Parameters() below
def double_Lorentz(x, a, b, A, w, x_0, A1, w1, x_01):
    return a * x + b + (2 * A / np.pi) * (w / (4 * (x - x_0) ** 2 + w ** 2)) + (2 * A1 / np.pi) * (
    w1 / (4 * (x - x_01) ** 2 + w1 ** 2))


class LineShapeParams():
    def __init__(self):
        self.line_shape_name = 'GL'
        self.names = []
        self.values = []
        self.bounds_min = []
        self.bounds_max = []
        self.description = []
        self.fixed_types = []

    def set_default_params(self):
        if self.line_shape_name == 'GL':
            self.names = ['m', 'F', 'E']
            self.values = [30, 0.5, 0.1]
            self.bounds_min = [0, 0, 0]
            self.bounds_max = [100, 10, 10]
            self.description = ['L = GL(100), G = GL(0)', 'FWHM', 'Central position']
            # do the optimizing search or not:
            self.fixed_types = ['yes', 'no', 'no']


class Spectra():
    # standart class for spectra
    def __init__(self):
        self.x = []
        self.y = []
        self.bg = []
        self.label = 'foo spectra'

    def plot(self):
        plt.plot(self.x, self.y, l=self.label)

    def plot_bg(self):
        plt.plot(self.x, self.bg, l='bg ['+self.label+']')


class SpectraConvolution():
    def __init__(self):
        self.src = Spectra()
        self.fit = Spectra()
        self.energyRegion = [704, 710]

        self.peak = {}
        self.struct_of_peaks = dict()
        self.loadData()

    def load_data(self):
        data = pickle.load(open('data.pkl', 'rb'))
        xData = data[0]
        yData = data[1]
        self.src.x = xData
        self.src.y = yData
        self.fit.x = xData
        self.fit.y = yData

    def select_region(self):
        idx = np.where((self.src.x >= np.min(self.energyRegion)) *
                       (self.src.x <= np.max(self.energyRegion)))
        self.fit.x = self.src.x[idx]
        self.fit.y = bg_move_curve_to_zero_line(self.src.y[idx])

    def subtract_bg(self):
        y = self.fit.y
        x = self.fit.x
        self.fit.y = bg_subtraction_recursively(x, y, iter=2)
        self.fit.bg = y - self.fit.y

    def add_peak(self):
        n = len(self.struct_of_peaks)
        peak = {}

        # configure peak:
        peak['id'] = n + 1
        peak['line shape'] = 'GL'
        par = LineShapeParams()
        par.line_shape_name = peak['line shape']
        par.set_default_params()
        peak['params'] = par
        peak['intensity'] = self.calc_peak_intensity(peak)

        # add peak tu structure:
        self.struct_of_peaks[n+1] = dict(self.peak)

    def calc_sum_of_peaks(self):
        # return sum of all added peaks
        n = len(self.struct_of_peaks)
        sum = 0
        for i in self.struct_of_peaks:
            current_peak = self.struct_of_peaks[i]
            sum = sum + self.calc_peak_intensity(current_peak)

        return sum

    def optimize_peak_params(self):
        pass

    def search_max_position(self):
        # return ebergy position of maxima intensity
        idx = np.where((self.fit.y == np.max(self.fit.y)))[0]
        return self.fit.x[idx]

    def calc_peak_intensity(self, selected_peak):
        # calc intensity from line shape name and selected energy region
        out = 0
        par = selected_peak['params']

        if selected_peak['line shape'] == 'GL':
            out = f_GL(x=self.fit.x, F=par.values[1], E=par.values[2], m=par.values[0])

        return out





# function for genetic algorithm to minimize (sum of squared error)
# bounds on parameters are set in generate_Initial_Parameters() below
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
    return np.sum((yData - double_Lorentz(xData, *parameterTuple)) ** 2)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)

    parameterBounds = []
    parameterBounds.append([-1.0, 1.0])  # parameter bounds for a
    parameterBounds.append([maxY / -2.0, maxY / 2.0])  # parameter bounds for b
    parameterBounds.append([0.0, maxY * 100.0])  # parameter bounds for A
    parameterBounds.append([0.0, maxY / 2.0])  # parameter bounds for w
    parameterBounds.append([minX, maxX])  # parameter bounds for x_0
    parameterBounds.append([0.0, maxY * 100.0])  # parameter bounds for A1
    parameterBounds.append([0.0, maxY / 2.0])  # parameter bounds for w1
    parameterBounds.append([minX, maxX])  # parameter bounds for x_01

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x


# load the pickled test data from original Raman spectroscopy
data = pickle.load(open('data.pkl', 'rb'))
xData = data[0]
yData = data[1]

# generate initial parameter values
initialParameters = generate_Initial_Parameters()

# curve fit the test data
fittedParameters, niepewnosci = curve_fit(double_Lorentz, xData, yData, initialParameters)

# create values for display of fitted peak function
a, b, A, w, x_0, A1, w1, x_01 = fittedParameters
y_fit = double_Lorentz(xData, a, b, A, w, x_0, A1, w1, x_01)

plt.plot(xData, yData)  # plot the raw data
plt.plot(xData, y_fit)  # plot the equation using the fitted parameters
plt.show()

print(fittedParameters)

if __name__ == '__main__':
    print('-> you run ', __file__, ' file in a main mode')