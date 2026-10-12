# Independent check of Lemma 1 (normalisation) by direct evaluation of the singular integral
import mpmath as mp
from core import lam
from run_all_helpers import tex_table, sci
mp.mp.dps=40
def lap_phi(n,s,x0):
    a=2*s; C=a*2**(a-1)*mp.gamma((1+a)/2)/(mp.sqrt(mp.pi)*mp.gamma(1-a/2))
    u=lambda x: (1-x**2)**s*mp.jacobi(n,s,s,x) if abs(x)<1 else mp.mpf(0)
    u2=mp.diff(u,x0,2)
    eps=mp.mpf(10)**(-6); d=(1-abs(x0))/2
    # subtract second-order Taylor term on (0,d): removes the t^{1-a} cancellation
    g=lambda t:(2*u(x0)-u(x0+t)-u(x0-t)+t**2*u2)/t**(1+a)
    I1=mp.quad(g,[eps,d/10,d])-u2*d**(2-a)/(2-a)
    h=lambda t:(2*u(x0)-u(x0+t)-u(x0-t))/t**(1+a)
    T=1+abs(x0)
    I2=mp.quad(h,sorted(set([d,1-abs(x0),T])))+2*u(x0)*T**(-a)/a
    return C*(I1+I2)
rows=[]; x0=mp.mpf('0.3')
for s in [0.25,0.5,0.75]:
    for n in [0,1,2,5]:
        v=float(lap_phi(n,s,x0)); ex=float(lam(n,s))*float(mp.jacobi(n,s,s,x0))
        rows.append([f"{s}",f"{n}",f"{v:.10f}",f"{ex:.10f}",sci(abs(v-ex)/abs(ex))]); print(rows[-1],flush=True)
tex_table("ex1_lemma",["$s$","$n$","$(-\\Delta)^s\\phi_n(0.3)$ (quadrature)","$\\lambda_nP_n(0.3)$","rel. diff."],rows,"cccrr")
