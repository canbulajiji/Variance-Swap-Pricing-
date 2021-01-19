#
# Valuation of Variance Swap
# incl. Vega function and implied volatility estimation
# -- class-based implementation
# VS_class.py
#

from cmath import log, sqrt, exp
from scipy import stats
import numpy as np
from scipy import linalg


class Stock(object):
    ''' Class for European call options in BSM model.
    
    Attributes
    ==========
    S0 : float
        initial stock/index level
    T : float
        maturity (in year fractions)
    r : float
        constant risk-free short rate
    y0 : float
        initial volatility
    a : float
        mean reversion rate
    b : float
        mean reversion value
    sigma : float
        volatility of volatility
    rho : correlation   
       
    Methods
    =======
    M - Explicit solution of M
    H
    R - without regime switching
    R_regime 
    
    '''
    
    def __init__(self, S0, y0, AF, r, sigma, a, b, T, rho):
        self.S0 = S0
        self.y0 = y0
        self.AF = AF
        self.r = r
        self.sigma = sigma
        self.a = a
        self.b = b
        self.rho = rho 
        self.delta = 1/self.AF
        self.T = T
        

            
    def M(self,w,t):
        ''' Returns M. '''
        A = -self.a+self.rho*self.sigma*(1j*w)
        B = sqrt(A**2+(self.sigma**2)*(1j*w)*(1j*w-1))
        C = (A+B)/(A-B)
        M = ((A+B)/(self.sigma**2))*(1-exp(B*(self.T-t)))/(1-C*exp(B*(self.T-t)))
        return M
    
#     def M(self,w,t):
#         ''' Returns M. '''
        
#         A = self.a-self.rho*self.sigma*(1j*w)
#         B = sqrt(A**2-(self.sigma**2)*(1j*w)*(1j*w-1))
#         C = (A-B)/(A+B)
#         M = ((A+B)/(self.sigma**2))*(C-C*exp(B*(self.T-t)))/(C-exp(B*(self.T-t)))
#         return M
    
#     def L(self,w,t):
#         A = self.a-self.rho*self.sigma*(1j*w)
#         B = sqrt(A**2+(self.sigma**2)*(1j*w)*(1j*w-1))
#         C = (A-B)/(A+B)
#         L = self.r*(w*1j-1)*(self.T-t)+((self.a*self.b)/self.sigma**2)*((A+B)*(self.T-t)-2*np.log((C-np.exp(B*(self.T-t)))/(C-1)))
#         return L
    def L(self,w,t):
        # L is the explicit integral without regime and jump
        A = -self.a+self.rho*self.sigma*(1j*w)
        B = sqrt(A**2-(self.sigma**2)*(1j*w)*(1j*w-1))
        C = (A+B)/(A-B)
        L = self.r*(w*1j-1)*(self.T-t)+((self.a*self.b)/self.sigma**2)*((A+B)*(self.T-t)-2*np.log((C-np.exp(B*(self.T-t)))/(C-1)))
        return L
    
    def H(self,t):
        ''' Returns H. '''
        c = (1-2*self.a/((self.sigma**2)*self.M(-2j,self.T-self.delta)))
        Nume = (2*self.a/self.sigma**2)*exp(-self.a*(self.T-self.delta-t))
        Deno = exp(-self.a*(self.T-self.delta-t))-c
        H = Nume/Deno
        return H
    
    def R(self):
        ''' Returns R=\int abH '''
        
        c = 2*self.a*self.b/self.sigma**2
        R = (c*log((exp(-self.a*self.T)-self.M(-2j,self.T-self.delta))
                   /(exp(-self.a*self.delta)-self.M(-2j,self.T-self.delta))))
        return R
    
    def R_regime(self,t):
        R = -self.a*self.b*Stock.H(t)
        return R
    
    def mean(self,t):
        #m = ((1-np.exp(-self.a*t))/(self.a*t))*self.y0+self.b*(1-((1-np.exp(-self.a*t))/(self.a*t)))
        m = self.b*((1-np.exp(-self.a*t)))/self.a
        return m
    
    
class Jump_Merton(Stock):
    '''Subclass for Merton jump diffusion. z~N(mu,sigma_J)
    
   Attributes
    ==========
    mu : float
        mean of jump size
    sigma_J : float
        sd of jump size 
    lambda_ : float
        jump intensity
    '''
    def __init__(self, S0, y0, AF, r, sigma, a, b, N, rho, mu,sigma_J,lambda_):
        super().__init__(S0, y0, AF, r, sigma, a, b, N, rho)
        self.mu = mu
        self.sigma_J = sigma_J
        self.lambda_ = lambda_
        
    def character(self,w):
        '''return characteristic function of Merton Jump'''
        
        phi = exp(1j*self.mu*w-0.5*w**2*self.sigma_J**2)
        return phi
    
    def mean(self):
        ''' return mean of exponential Merton jump'''
        
        mean = exp(self.mu+self.sigma_J**2/2);
        return mean
    
#     def L(self,w,t):
#         phi = self.character(w)
#         L = (self.r-self.lambda_*self.mean())*1j*w-(self.r+self.lambda_)+self.lambda_*phi+self.a*self.b*self.M(w,t)
#         return -L
    
    def L(self,w,t):
        phi = self.character(w)
        L = (self.r-self.lambda_*self.mean())*1j*w-(self.lambda_+self.r)+self.lambda_*phi+self.a*self.b*self.M(w,t)
        return L
    
    def mean_com(self):
        m = self.lambda_*(self.mu**2+self.sigma_J**2)
        return m
    
class Jump_Kou(Stock):
    '''Subclass for Kou jump diffusion with jump size satisfy double exponential distribution
    
   Attributes
    ==========
    eta1 : float
        parameter of 1st exponential
    eta2 : float
        parameter of 2nd exponential 
       p : probability of the occurences of the 1st exponential
    lambda_ : float
        jump intensity
    '''
    
    def __init__(self, S0, y0, AF, r, sigma, a, b, N, rho, eta1, eta2, p, lambda_):
        super().__init__(S0, y0, AF, r, sigma, a, b, N, rho)
        self.eta1 = eta1
        self.eta2 = eta2
        self.p = p
        self.lambda_ = lambda_
        
    def character(self,w):
        '''return characteristic function of Kou Jump'''
        
        phi = self.p*self.eta1/(self.eta1-1j*w)+(1-self.p)*self.eta2/(self.eta2+1j*w);
        return phi
    
    def mean(self):
        ''' return mean of exponential Kou jump'''
        
        mean = (1-self.p)/(self.eta2+1)-self.p/(self.eta1-1);
        return mean
    
    def L(self,w,t):
        phi = self.character(w)
        L = ((self.r-self.lambda_*self.mean())*1j*w-(self.r+self.lambda_)+self.lambda_*phi+self.a*self.b*self.M(w,t))
        return L
    
    
class Regime2(Stock):
    '''Subclass for Regime Switching model 
       
        Attributes
        ==========
           Q is the transition matrix

        ''' 
           
    def __init__(self, Q):
       
        self.Q = Q  
        
        
    def character(self,v,t,T):
        E = np.ones(self.Q.ndim)
        phi = linalg.expm(self.Q.transpose()*(T-t)+v)
        Phi = np.matmul(phi，E)
        return phi, Phi
      
 
    
                        
                        
                        