'''
get it from the:
https://github.com/aaronth/wraith
'''

# from pylab import *
from scipy.special import wofz
import numpy as np

###########################################
# Functions to feed into fitting routines
###########################################

def analyzer_function(p, E):
    A, B = p
    BG = A/np.sqrt(abs(B-E))
    #print "A: %f, B: %f"%(A,B)
    return BG

def analyzer_function_2(p, E):
    A, B = p
    BG = (A*E)**2/((B-E)**3)
    #print "A: %f, B: %f"%(A,B)
    # BG = BG * (BG > 0)
    return BG

def slope(p, E):
    m, b = p
    BG = b + m * E
    return BG

def parabola(p, E):
    a, b, c= p
    BG = a* E**2 + b*E + c
    return BG

def cubic_parabola(p, E):
    a, b, c, d= p
    BG = a* E**3 + b*E**2 + c*E + d
    return BG

def sigmoid(p,x):
    x0,y0,c,k=p
    y = c / (1 + np.exp(-k*(x-x0))) + y0
    return y

def fitting_arctan(p, E):
    A, B, u, b = p
    BG = b + A * np.arctan( B * (E - u) )
    return BG

def sloped_arctan(p, E):
    A, B, u, m, b = p
    BG = b + m * E + A * np.arctan( B * (E - u) )
    return BG
sloped_arctan.latex = r'$s_arctan(E) = b + mE + A {\rm arctan} ( B (E - u) )$'

def tougaard(p, E):
    # change Tougaard abstract background to physical background E_loss, R_loss
    # E_loss is mean loss in eV, ~ 10 - 80, Tougaard best 63.6
    # R_loss is ratio zero loss to single loss, ~ 0.8 to 1.0 Tougaard 0.872
    # B*T/(C+T**2)**2
    # C -> 4*dE_ave/pi**2
    # B -> 8*R_loss*dE_ave
    return K(p, E)

def tougaard_best(p, E):
    return tougaard(0.872, 63.6, E)

def K(p, E):
    """Physical convolution kernel for Tougaard background"""
    R_loss, E_loss = p
    K_ = (8.0/np.pi**2)*R_loss*E_loss**2 * E / ((2.0*E_loss/np.pi)**2 + E**2)**2
    """convolution kernel for Tougaard background"""
    #B, C = p
    #K_ = B * E / (C + E**2)**2
    K_ = K_*(K_>0)
    return K_

def K3(p, E):
    """convolution kernel for Tougaard background"""
    B, C, D = p
    K_ = B * E / ((C + E**2)**2 + D*E**2)
    K_ = K_*(K_>0)
    return K_

def voigt(p, E):
    """
        the voigt function = convolve(gaussian, lorentzian)
        p = A, mu, sigma
        """
    alpha, gamma = p
    dE = E[1]-E[0]
    # return dE * convolve( lorentzian(p,E), gaussian(p,E), 'same' )
    return V(E, alpha, gamma)

def spin_split_gl(params, E):
    """
        spin split for the gl function
        a, mu, sigma are for peak1
        ratio_a * a == a for peak2
        ratio_sigma * sigma = sigma for peak2
        ratio_area == ratio_a * ratio_sigma can be fixed using a boundary penalty
        ratio_area -> 1/2 for p orbitals
        ratio_area -> 2/3 for d orbitals
        ratio_area -> 3/4 for f orbitals
        split is the spacing between peak1 and peak2
        """
    a, mu, sigma, m, ratio_a, ratio_area, split = params
    ratio_sigma = ratio_area / ratio_a
    return gl_(a, mu, sigma, m, E) + gl_(ratio_a * a, mu - split, ratio_sigma * sigma, m, E)
spin_split_gl.latex = r'$ssgl(E) = gl(A,\mu,\sigma,m,E) + gl(R_A A, \mu + \Delta_E, \frac{R_{Area}}{R_A} \sigma, E)$'

def gl_(a, mu, sigma, m, E):
    return a * np.exp(-2.772589 * (1 - m) * (E - mu)**2/sigma**2) / (1 + 4 * m * (E - mu)**2/sigma**2)

def gl(params, E):
    a, mu, sigma, m = params
    return gl_(a, mu, sigma, m, E)
gl.latex = r'$gl(A,\mu,\sigma,m,E) = A e^{\frac{(-4ln(2) (1-m) (E - \mu)^2/\sigma^2)}{(1+4m(E-\mu)^2/\sigma^2)}}$'

def gl50(params, E):
    a, mu, sigma = params
    m = 0.5
    return gl_(a, mu, sigma, m, E)

def gls_(a, mu, sigma, m, E):
    return a * (1 - m) * np.exp(-2.772589 * (E - mu)**2/sigma**2) + m/(1 + 4 * (E - mu)**2/sigma**2)

def gls(params, E):
    a, mu, sigma, m = params
    return gls_(a, mu, sigma, m, E)

def lorentzian_(a, mu, sigma, E):
    return a * 1/(1 + ((E - mu)/sigma)**2)

def lorentzian(params, E):
    a, mu, sigma = params
    return lorentzian_(a, mu, sigma, E)

def gaussian_(a, mu, sigma, E):
    return a * np.exp(-((E-mu)/sigma)**2)

def gaussian(params, E):
    a, mu, sigma = params
    return gaussian_(a, mu, sigma, E)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

#####################################################
# Penalty functions
#
# Returns a scale factor for residuals depending on
# the relationship between the parameter value and
# a declared range
#####################################################

#Residuals are unscaled regardless of range
def no_penalty(range, p):
    return 1.0

#Residuals are scaled by a constant factor of 100 if parameter value is outside of the range
def notch_penalty(range, p):
    value = 1.0
    if p < range[0] or p > range[1]:
        value = 100.0
    return value

#Residuals are scaled by an exponentially growing factor if parameter is outside of the range
def exp_penalty(range, p):
    A = 1.0/np.diff(range)
    value = 1.0
    if p < range[0]:
        value = 1.0 + np.exp(A*((range[0]+0.05*np.diff(range)) - p ))
    elif p > range[1]:
        value = 1.0 + np.exp(A*(p - (range[1]-0.05*np.diff(range)) ))
    return value

#Residuals are scaled by a quadratically growing factor if parameter value is outside of the range
def quad_penalty(range, p):
    A = 100.0/np.diff(range)
    value = 1.0
    lower_bound = (range[0]+0.02*np.diff(range))
    upper_bound = (range[1]-0.02*np.diff(range))
    if p < lower_bound:
        value = 1.0 + A*(lower_bound - p )**2
    elif p > upper_bound:
        value = 1.0 + A*(p - upper_bound )**2
    return value
