import numpy as np
from scipy.special import iv
from scipy.special import kv
import scipy
import mpfit

def v_circ_exp(xkpc,param,arrsize=1500.):
    '''
    function for a rotation curve 
    that turns over and declines 
    at large radii
    '''
    # exponential disk model velocity curve (Freeman 1970; Equation 10)
    # v^2 = R^2*!PI*G*nu0*a*(I0K0-I1K1)
    
    # param = [r0,s0,v0,roff,theta]
    # r0 = 1/a = disk radii
    # R = radial distance 
    # roff = offset of velocity curve
    # from 0 -> might want to set to fixed at 0?)
    # s0 = nu0 = surface density constant (nu(R) = nu0*exp(-aR))
    # v0 is the overall velocity offset
    
    # G
    G = 6.67408e-11 #m*kg^-1*(m/s)^2
    G = G*1.989e30  #m*Msol^-1*(m/s)^2
    G = G/3.0857e19 #kpc*Msol^-1(m/s)^2
    G = G/1000./1000.
    
    # parameters
    r0  = param[0]
    s0  = 10**np.double(param[1])
    v0  = param[2]
    roff = param[3]
    theta =param[4]
    
    # evaluate bessel functions (evaluated at 0.5aR; see Freeman70)
    rr = 0.025*np.arange(arrsize)+0.001 # set up an array
    #rr = 0.025*np.arange(15000.)+0.001
    temp=(0.5*(rr)/r0)
    temp[temp>709.]=709.
    I0K0 = iv(0,temp)*kv(0,temp)
    I1K1 = iv(1,temp)*kv(1,temp)
    bsl  = I0K0 - I1K1
    
    # velocity curve
    v2a  =  rr*((np.pi*G*s0*bsl)/r0)**0.5  
    v2a  = v2a*np.sin(np.pi*theta/180.)
    
    # reflect the rotation curve
    rr_r = np.append(-np.array(rr),np.array(rr))
    v2_r = np.append(-np.array(v2a),np.array(v2a))
    rrb   = np.sort(rr_r) #rr_r(sort(rr_r))
    v2b   = v2_r[np.argsort(rr_r)] #v2_r(sort(rr_r))
    
    
    # regrid back onto kpc scale and velocity offset
    f=scipy.interpolate.interp1d(rrb,v2b,bounds_error=False)
    v2 = f(xkpc-roff)+v0
    
    return v2

def fit_v_circ_exp(r,v,dv,p0,w=None,rob_errs=True,fixed=0):
    '''
    function to fit an exponential disk 
    model rotation curve to data
    
    r: radius
    v: velocity
    dv: velocity uncertainty
    p0: initial guess for parameters [r0,s0,v0,roff,theta]
    '''
    
    #if there are no weights
    if w==None:
        w=np.zeros(len(r))+1.
    
    #==========================================================================
    # fit the rotation curve
    #==========================================================================
    
    #first set up mpfit
    
    #define figure of merit
    def myfunct(p,fjac=None,x=None,y=None,dy=None,weights=None):
        #function to return the weighted deviates
        model = v_circ_exp(x,p)
        status=0
        error=dy
        return([status,weights*(y-model)/error])

    #set up priors array       
    parinfo=[{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'tied':''} for i in range(5)]
    
    #set prior variable limits for fitting
    #r0,s0,v0,roff,theta
    parinfo[0]['limited'][0]=1
    parinfo[0]['limited'][1]=0
    parinfo[0]['limits'][0]=0.
    parinfo[0]['limits'][1]=4.
    parinfo[1]['limited'][0]=0
    parinfo[1]['limited'][1]=0
    parinfo[1]['limits'][0]=0
    parinfo[1]['limits'][1]=0
    parinfo[2]['limited'][0]=1
    parinfo[2]['limited'][1]=1
    parinfo[2]['limits'][0]=-550.
    parinfo[2]['limits'][1]=550.
    parinfo[3]['limited'][0]=1
    parinfo[3]['limited'][1]=1
    parinfo[3]['limits'][0]=-0.5
    parinfo[3]['limits'][1]=.5
    parinfo[4]['limited'][0]=1
    parinfo[4]['limited'][1]=1
    parinfo[4]['limits'][0]=0.
    parinfo[4]['limits'][1]=90.

    parinfo[2]['fixed']=fixed
    parinfo[3]['fixed']=fixed   
    parinfo[4]['fixed']=fixed    


    #rest of set up for mpfit
    fa={'x':r, 'y':v, 'dy':dv,'weights':w}            
    #do the fit
    m=mpfit.mpfit(myfunct,p0,functkw=fa,parinfo=parinfo,quiet=1)
    parinfo[4]['fixed']=1. #fix the inclination to initial best fit
    m=mpfit.mpfit(myfunct,m.params,functkw=fa,parinfo=parinfo,quiet=1)
    pars=m.params.copy() #first save the parameters
    try:
        error=m.perror.copy() #and error
    except:
        error=m.perror
    if rob_errs==True:
        try:
            print('calculating errors...')
            #refit holding other parameters to get reasonable errors
            for i in range(len(pars)-1):
                for j in range(len(pars)-1):
                    if j!=i:
                        parinfo[j]['fixed']=1
                m=mpfit.mpfit(myfunct,pars,functkw=fa,parinfo=parinfo,quiet=1)
                error[i]=m.perror[i]
                for j in range(len(pars)-1):
                    parinfo[j]['fixed']=0
        except:
            error=error
    #get the normalised errors 
    DOF=len(r)-len(p0)
    try:
        error=np.sqrt(m.fnorm/DOF)*error
    except:
        error=error
    
    return [m.params,error,v_circ_exp(r,m.params)]
