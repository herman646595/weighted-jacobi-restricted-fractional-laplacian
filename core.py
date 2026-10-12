import numpy as np, mpmath as mp
from scipy.special import gammaln, roots_jacobi, eval_jacobi, gamma
from scipy.linalg import eigh

def lam(n,s): n=np.asarray(n,float); return np.exp(gammaln(n+2*s+1)-gammaln(n+1))
def hn(n,s):
    n=np.asarray(n,float)
    return np.exp((2*s+1)*np.log(2)+2*gammaln(n+s+1)-np.log(2*n+2*s+1)-gammaln(n+2*s+1)-gammaln(n+1))
def lh(n,s):
    n=np.asarray(n,float)
    return np.exp((2*s+1)*np.log(2)-np.log(2*n+2*s+1)+2*(gammaln(n+s+1)-gammaln(n+1)))

def gj(Q,a): x,w=roots_jacobi(Q,a,a); return x,w
def Pmat(N,s,x): return np.array([eval_jacobi(n,s,s,x) for n in range(N+1)])

def load_coeffs(f,N,s,Q):
    """b_m=int f w_s P_m dx via Gauss-Jacobi(s,s) with Q nodes; returns c_m=b_m/(lam h)"""
    x,w=gj(Q,s); P=Pmat(N,s,x); b=P@(w*f(x)); n=np.arange(N+1)
    return b/(lam(n,s)*hn(n,s)), b

def c_closed(m,s,p):
    """closed-form even coefficient c_{2m}(u_p), Theorem (b); returns float (mpmath internally)"""
    q=p-s; out=[]
    for mm in np.atleast_1d(m):
        mm=int(mm)
        v=-mp.sin(mp.pi*q)*mp.gamma(p+1)*mp.gamma(q+1)/(2*mp.pi**1.5)*(4*mm+2*s+1)*mp.mpf(2)**(2*mm)\
          *mp.gamma(mm+s+.5)*mp.gamma(mm+.5)*mp.gamma(mm-q)/(mp.gamma(2*mm+s+1)*mp.gamma(mm+p+1.5))
        out.append(float(v))
    return np.array(out)

def coeffs_up(p,s,nmax):
    """all Jacobi coefficients c_n(u_p), n<=nmax, by closed form (non-integer q) """
    c=np.zeros(nmax+1); m=np.arange(0,nmax//2+1); c[0::2]=c_closed(m,s,p)[:len(c[0::2])]; return c

def c_quad(p,s,nmax,Q=400):
    x,w=gj(Q,p); P=Pmat(nmax,s,x); n=np.arange(nmax+1)
    return (P@w)/hn(n,s)

def c_closed_fast(nmax,s,p):
    """vectorised log-gamma version (sign of Gamma(m-q) handled)"""
    q=p-s; m=np.arange(0,nmax//2+1,dtype=float)
    lg=gammaln
    sg=np.where(m-q>0,1.0,np.sign(gamma(np.where(m-q>0,1.0,m-q))))
    lv=(np.log(abs(np.sin(np.pi*q)))+lg(p+1)+lg(q+1)-np.log(2)-1.5*np.log(np.pi)+np.log(4*m+2*s+1)
        +2*m*np.log(2)+lg(m+s+.5)+lg(m+.5)+lg(m-q+(m-q<=0)*0)-lg(2*m+s+1)-lg(m+p+1.5))
    # for m<=q use mpmath-exact values to avoid gammaln(neg)
    c=-np.sign(np.sin(np.pi*q))*sg*np.exp(lv)
    k=int(np.floor(q))+2
    c[:k]=c_closed(m[:k],s,p)
    out=np.zeros(nmax+1); out[0::2]=c; return out

def u_N(x,c,s):
    P=Pmat(len(c)-1,s,x); return (1-x**2)**s*(c@P)

def errors(uex,c,s,Lq=3000,Mgrid=20001):
    xq,wq=np.polynomial.legendre.leggauss(Lq)
    e=uex(xq)-u_N(xq,c,s); L2=np.sqrt(np.sum(wq*e**2))
    xg=np.linspace(-1,1,Mgrid); Linf=np.max(abs(uex(xg)-u_N(xg,c,s)))
    return L2,Linf
