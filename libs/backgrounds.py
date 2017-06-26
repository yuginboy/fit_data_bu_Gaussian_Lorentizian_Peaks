'''
Collections of the different BackGround calculations
'''
from numpy import array, linspace, arange, zeros, ceil, amax, amin, argmax, argmin, abs, trapz
from numpy import polyfit, polyval, seterr, trunc, mean
from numpy.linalg import norm
from scipy.interpolate import interp1d
import numpy as np
import scipy.optimize as sp
from libs.fitting_functions import *
import os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess



DEBUG = False
OPTION = 2

def moving_average(a, windowsize=3) :
    n = windowsize
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def shirley_base(x, y, tol=1e-5, maxit=5):
    """ S = shirley_base(x,y, tol=1e-5, maxit=10)
    Calculate the best auto-Shirley background S for a dataset (x,y). Finds the biggest peak
    and then uses the minimum value either side of this peak as the terminal points of the
    Shirley background.
    The tolerance sets the convergence criterion, maxit sets the maximum number
    of iterations.
    https://github.com/kaneod/physics/tree/master/python
    """

    # Make sure we've been passed arrays and not lists.
    x = array(x)
    y = array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print ("shirley_base: One of the arrays x or y is empty. Returning zero background.")
        return zeros(x.shape)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = abs(y - amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print ("shirley_base: Boundaries too high for algorithm: returning a zero background.")
        return zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - amin(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - amin(y[maxidx:])).argmin() + maxidx
    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    it = 0
    while it < maxit:
        if DEBUG:
            print ("Shirley iteration: ", it)
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] +
                                                   y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print ("shirley_base: Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B


def shirley_new(x, y_in, maxit=5, numpoints=3):
    '''
    :param x: array of X
    :param y: array of Intensity or Y
    :param maxit: max number of iterations. Default is 5
    :param numpoints: number of points to calculate the average value for Y shift
    :return: Backgraund (Shirley-type ) y-values
    '''
    # Make sure we've been passed arrays and not lists.
    x = array(x)
    y = y_in.copy() # to avoid changes in the source array
    y = array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print ("shirley_new: One of the arrays x or y is empty. Returning zero background.")
        return zeros(x.shape)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = abs(y - amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print ("shirley_base: Boundaries too high for algorithm: returning a zero background.")
        return zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - amin(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - amin(y[maxidx:])).argmin() + maxidx
    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    zz = B = zeros(x.shape)
    B[:lmidx] = yl - yr

    it = 0
    for it in range(maxit):
        for i in range(len(y)):
            zz = y - yr - B
            Integ_k = trapz(zz, x=x)
            zz[0:i] = 0
            Integ = trapz(zz, x=x)
            kn = (yl - yr)/Integ_k
            B[i] = kn * Integ


    # Calculate a Y shift:
    if (rmidx+numpoints) > len(y):
        delta = y[rmidx]
    else:
        delta = y[rmidx:rmidx+numpoints].mean()

    # output
    if is_reversed:
        #return (yr + B)[::-1]
        return (delta + B)[::-1]
    else:
        # return yr + B
        return delta + B

def bg_move_curve_to_zero_line(y):
    min_dy = np.min(np.abs(y))
    idx = np.where(np.abs(y) == min_dy)
    dy = y[idx][0]
    return y - dy

def bg_subtraction_recursively (x, y, iter=2, numpoints = 1):
    if iter <= 0:
        return y - shirley_new(x, y, numpoints=numpoints)
    # print('===== iter = {}\n'.format(iter))
    return bg_subtraction_recursively(x, bg_subtraction_recursively(x, y, iter=0, numpoints=1), iter - 1, numpoints=numpoints)

#############################################################
# Fitting machinery
#   Penalty: a class to steer fitting
#   Background: generic convolution based background
#############################################################

class Penalty:
    """Encapsulates a penalty function to steer fitting for a parameter"""

    def __init__(self, f_range, f):
        """Initialize penalty function"""
        self.range = f_range
        self.f = f

    def __call__(self, p):
        """Penalty!"""
        return self.f(self.range, p)


class Background():
    def __init__(self, spectrum, name='tougaard',
                 variables=['ratio', 'ave dE'],
                 values=np.r_[100, 100],
                 penalties=[Penalty(np.r_[0, 100], no_penalty), Penalty(np.r_[0, 100], no_penalty)],
                 kernel=K,
                 kernel_end=200):
        self.spectrum = spectrum
        self.name = name#
        self.values = values
        self.variables = variables
        self.penalties = penalties
        self.kernel = kernel
        self.kernel_end = kernel_end

    def set_spec(self, spec):
        self.penalties = []
        for range in spec['ranges']:
            self.penalties.append(Penalty(range, eval(spec['penalty_function'])))
        self.name = spec['name']
        self.variables = spec['variables']
        self.values = spec['values']
        self.kernel = eval(spec['function'])

    def get_spec(self):
        ranges = []
        for pen in self.penalties:
            ranges.append(pen.range)

        bg_spec = {'name': self.name,
                   'function': self.kernel.func_name,
                   'penalty_function': self.penalties[0].f.func_name,
                   'variables': self.variables,
                   'values': self.values,
                   'ranges': ranges
                   }
        return bg_spec

    def f(self, values, E, spectrum):
        dE = E[1] - E[0]
        #spectrum = spectrum - min(spectrum)
        spectrum = spectrum - spectrum[-1]
        #set highest energy to zero, respect physical model behind Tougaard
        bg = dE * np.convolve( spectrum, self.kernel( values, E)[::-1], 'full')
        return bg[bg.size-spectrum.size:]

    def residuals(self, values, E, spectrum):
        res = spectrum - self.f(values, E, spectrum)
        i = 0
        for p in values:
            res *= self.penalties[i](p)
            i += 1

        res[res<0] = res[res<0]*20
        return res

    def EE(self, dE):
        return np.arange(0, self.kernel_end, abs(dE))

    def optimize_fit(self, E, spectrum):
        #offset = min(spectrum)
        offset = spectrum[-1]
        #set highest energy to zero, respect physical model behind Tougaard
        spectrum = spectrum - offset
        self.dE = E[1]-E[0]
        plsq = sp.leastsq(self.residuals, self.values, args=(self.EE(self.dE), spectrum))
        self.values = plsq[0]
        return self.f(self.values, self.EE(self.dE), spectrum) + offset

    def __call__(self, E, spectrum):
        #offset = min(spectrum)
        offset = spectrum[-1]
        spectrum = spectrum - offset
        self.dE = E[1]-E[0]
        return self.f(self.values, self.EE(self.dE), spectrum) + offset


def bg_subtraction (x, y):
    y = bg_move_curve_to_zero_line(y)
    # y = y - moving_average(y, windowsize=10)
    # result = lowess(y, x, frac=0.7)
    #
    # y_smooth = result[:,1]
    # plt.plot(x,y, x, y_smooth )
    # plt.show()
    res = y - shirley_new(x, y, numpoints=1)
    # print('===== iter = {}\n'.format(iter))
    return res

class BackgroundByName():
    global_optimalValues = []
    def __init__(self):
        self.name = 'tougaard'
        self.x = []
        self.y = []
        self.values = []
        self.optimal_values = []


    def calc_BG(self):
        f_penalty = no_penalty

        kernelname = self.name

        if kernelname == 'analyzer_function':
            values = np.r_[1, 6]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        if kernelname == 'analyzer_function_2':
            values = np.r_[100, 100,]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         ]
        if kernelname == 'slope':
            values = np.r_[100, 100]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        if kernelname == 'parabola':
            values = np.r_[100, 100, 100]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'cubic_parabola':
            values = np.r_[100, 100, 10, 10]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'sigmoid':
            values = np.r_[1, 1, 10, 1]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'fitting_arctan':
            values = np.r_[100, 100, 10, 10]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'sloped_arctan':
            values = np.r_[100, 100, 10, 10, 10]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'K3':
            values = np.r_[100, 100, 10,]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty),
                         Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'voigt':
            values = np.r_[0.1, 0.1]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'toguaard' or kernelname == 'K':
            values = np.r_[100, 100]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        elif kernelname == 'K3':
            values = np.r_[100, 100]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]
        else:
            kernelname = 'tougaard_best'
            values = np.r_[0.1, 0.1]
            penalties = [Penalty(np.r_[0, 100], f_penalty), Penalty(np.r_[0, 100], f_penalty)
                         ]

        self.bg = Background(self.y,
                             kernel=eval(kernelname),
                             values=values,
                             penalties=penalties
                           )
    def find_optimal_values(self):
        self.bg.optimize_fit()
        self.optimal_values = self.bg.values

    def set_values_to_optimal(self):
        self.values = self.optimal_values

    def set_global_optimalValues(self):
        BackgroundByName.global_optimalValues = self.optimal_values

    def get_global_optimalValues(self):
        self.optimal_values = BackgroundByName.global_optimalValues

def plot_crv(data):
        x, y = data[0, :], data[1, :]
        idx = np.where((x >= np.min(energyRegion)) *
                       (x <= np.max(energyRegion)))
        xx = x[idx]
        yy = bg_move_curve_to_zero_line(y[idx])
        yy = yy / np.max(yy)
        plt.plot(xx, yy, 'o-', label='raw')
        plt.axhline(0, color='k')

        y_shir_bg = shirley_new(xx, yy, numpoints=1)
        plt.plot(xx, y_shir_bg, label='shirley BG')
        # y1 = yy - y_shir_bg
        # plt.plot(xx, y1, label='shirley 1')

        # y_shir_bg2 = shirley_new(xx, y1, numpoints=1)
        # plt.plot(xx, y_shir_bg2, label='shirley BG2')
        # y2 = y1 - y_shir_bg2
        # plt.plot(xx, y2, label='shirley 2')
        plt.plot(xx, bg_subtraction_recursively(xx, yy, iter=2), label='shirley recurs')

if __name__=='__main__':
    print('-> you run ', __file__, ' file in a main mode')
    from libs.dir_and_file_operations import runningScriptDir

    experimentDataPath = os.path.join(runningScriptDir, 'data', 'model_with_Au','raw')
    experiment_filename = r'raw_Co2p_alpha=0deg.txt'
    energyRegion = [704, 710]
    data = np.loadtxt(os.path.join(experimentDataPath, experiment_filename), unpack=True)
    plot_crv(data)
    # data = np.loadtxt(r'/home/yugin/VirtualboxShare/Co-CoO/out/00011/out/Co2p.dat', unpack=True)
    # plot_crv(data)



    # bg = BackgroundByName()
    # bg.y = yy
    # bg.x = xx
    # bg.calc_BG()
    # y2_bg = bg.y
    # plt.plot(xx, y2_bg, label='BG')


    plt.legend(loc='best')
    plt.show()
    print('stop debug of ', __file__)