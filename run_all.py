import numpy as np, mpmath as mp, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import *
plt.rcParams.update({"font.size":10,"figure.dpi":130,"axes.grid":True,"grid.alpha":.3})
os.makedirs("figs",exist_ok=True); os.makedirs("tables",exist_ok=True)
R={}
def sci(x): 
    if x==0: return "0"
    e=int(np.floor(np.log10(abs(x)))); m=x/10**e; return f"${m:.2f}\\times10^{{{e}}}$"
def tex_table(fn,header,rows,colfmt):
    with open(f"tables/{fn}.tex","w") as f:
        f.write("\\begin{tabular}{"+colfmt+"}\n\\toprule\n"+" & ".join(header)+" \\\\\n\\midrule\n")
        for r in rows: f.write(" & ".join(r)+" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
def energy_err(c_num,s,c_ex_full):
    """||u-u_N||_a^2 = sum_{n<=N} lh (c_num-c_ex)^2 + sum_{n>N} lh c_ex^2 ; c_ex_full has length M+1>N"""
    N=len(c_num)-1; n=np.arange(len(c_ex_full)); l=lh(n,s)
    d=c_ex_full.copy(); d[:N+1]-=c_num
    return np.sqrt(np.sum(l*d**2))

# ---------------- Example 1: torsion + Lemma check ----------------
print("EX1")
R['lemma']='see lemma_check.py'
rows=[];R["ex1"]=[]
for s in [0.1,0.25,0.5,0.75,0.9]:
    out=[]
    for N in [0,8,32]:
        c,_=load_coeffs(lambda x:np.ones_like(x),N,s,Q=N+8)
        cex=np.zeros(N+1); cex[0]=1/float(gamma(2*s+1))
        E=energy_err(c,s,cex)
        uex=lambda x:(1-x**2)**s/float(gamma(2*s+1))
        L2,Li=errors(uex,c,s)
        out+= [E,L2,Li]
    rows.append([f"{s}"]+[sci(v) if v>0 else "0" for v in out[:3]]+[sci(v) for v in out[3:]])
    R["ex1"].append((s,out))
tex_table("ex1_torsion",["$s$","$E_0$","$L^2_0$","$L^\\infty_0$","$E_{32}$","$L^2_{32}$","$L^\\infty_{32}$"],
          [[r[0],r[1],r[2],r[3],r[-3],r[-2],r[-1]] for r in rows],"crrrrrr")
fig,ax=plt.subplots(1,2,figsize=(9,3.4)); x=np.linspace(-1,1,400)
for s in [0.1,0.25,0.5,0.75,0.9]: ax[0].plot(x,(1-x**2)**s/float(gamma(2*s+1)),label=f"s={s}")
ax[0].set_title("Torsion solution $u=(1-x^2)^s/\\Gamma(2s+1)$"); ax[0].legend(fontsize=7); ax[0].set_xlabel("x")
n=np.arange(0,201)
for s in [0.1,0.25,0.5,0.75,0.9]: ax[1].loglog(n+1,lh(n,s),label=f"s={s}")
ax[1].set_title("Diagonal entries $\\lambda_nh_n\\asymp (n+1)^{2s-1}$"); ax[1].set_xlabel("n+1"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/ex1.png"); plt.close()

# ---------------- Example 2: integer q, f_p with GJ(s,s) quadrature ----------------
print("EX2")
mp.mp.dps=30
def fp_nodes(s,p,x):
    pref=mp.mpf(2)**(2*s)*mp.gamma(s+.5)*mp.gamma(p+1)/(mp.sqrt(mp.pi)*mp.gamma(p+1-s))
    return np.array([float(pref*mp.hyp2f1(s+.5,s-p,.5,xi**2)) for xi in x])
cases=[(0.5,1.5),(0.5,2.5),(0.3,2.3),(0.7,3.7)]; R["ex2"]={}
fig,ax=plt.subplots(figsize=(5.5,3.8)); rows=[]
for (s,p) in cases:
    q=int(round(p-s)); x,w=gj(40,s); fx=fp_nodes(s,p,x); P=Pmat(10,s,x); b=P@(w*fx); n=np.arange(11)
    c=b/(lam(n,s)*hn(n,s)); cex=c_quad(p,s,10,Q=60)
    errs=[]
    for N in range(0,9):
        d=cex.copy(); d[:N+1]-=c[:N+1]; errs.append(np.sqrt(np.sum(lh(n,s)*d**2)))
    # E for N>2q: exact zero tail; floor reflects roundoff
    R["ex2"][f"{s},{p}"]=errs
    ax.semilogy(range(9),np.maximum(errs,1e-17),"o-",label=f"s={s}, p={p} (q={q})")
    rows.append([f"{s}",f"{p}",f"{q}"]+[sci(errs[N]) if errs[N]>1e-16 else "$<10^{-16}$" for N in [0,1,2,3,4,6,8]])
ax.set_xlabel("N"); ax.set_ylabel("energy error"); ax.legend(fontsize=7); ax.set_title("Integer $q$: finite termination at $N=2q$")
plt.tight_layout(); plt.savefig("figs/ex2.png"); plt.close()
tex_table("ex2",["$s$","$p$","$q$","$N{=}0$","$N{=}1$","$N{=}2$","$N{=}3$","$N{=}4$","$N{=}6$","$N{=}8$"],rows,"ccc"+"r"*7)

# ---------------- Example 3: non-integer q ----------------
print("EX3")
cases=[(0.25,1.0),(0.5,1.0),(0.75,1.0),(0.5,0.7)]; Ns=[2,4,8,16,32,64,128,256]; M=400000
R["ex3"]={}; rows_rate=[]; fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for (s,p) in cases:
    q=p-s; cfull=c_closed_fast(M,s,p); n=np.arange(M+1); l=lh(n,s)
    uex=lambda x,p=p:(1-x**2)**p
    E=[];L2=[];Li=[]
    for N in Ns:
        E.append(np.sqrt(np.sum((l*cfull**2)[N+1:])))
        if N<=128: a,b=errors(uex,cfull[:N+1],s)
        else: a,b=np.nan,np.nan
        L2.append(a);Li.append(b)
    r=2*q+s+1; A=2**(1-s)*abs(np.sin(np.pi*q))*float(gamma(p+1)*gamma(q+1))/np.pi
    Cth=2**(s+r)*A/np.sqrt(2*(4*q+2))
    eoc=[np.log(E[i]/E[i+1])/np.log((Ns[i+1]+1)/(Ns[i]+1)) for i in range(len(Ns)-1)]
    ratio=[E[i]*Ns[i]**(2*q+1)/Cth for i in range(len(Ns))]
    R["ex3"][f"{s},{p}"]=dict(N=Ns,E=E,L2=L2,Li=Li,eoc=eoc,Cth=Cth,ratio=ratio,q=q)
    ax[0].loglog(Ns,E,"o-",label=f"s={s},p={p}"); ax[0].loglog(Ns,[Cth*N**(-(2*q+1)) for N in Ns],"k:",lw=.8)
    ax[1].loglog(Ns[:-1],L2[:-1],"o-",label=f"s={s},p={p}")
    ax[2].loglog(Ns[:-1],Li[:-1],"o-",label=f"s={s},p={p}")
    rows_rate.append([f"{s}",f"{p}",f"{q:.2f}",f"{2*q+1:.2f}",f"{eoc[-1]:.3f}",f"{Cth:.4f}",f"{ratio[-1]:.4f}"])
    # table per case
    rows=[[str(N),sci(E[i]),(f"{eoc[i-1]:.3f}" if i>0 else "--"),sci(L2[i]) if N<=128 else "--",sci(Li[i]) if N<=128 else "--",f"{ratio[i]:.4f}"] for i,N in enumerate(Ns)]
    tex_table(f"ex3_s{s}_p{p}".replace(".",""),["$N$","$\\|u-u_N\\|_a$","EOC","$\\|u-u_N\\|_{L^2}$","$\\|u-u_N\\|_{\\infty}$ (sampled)","$N^{2q+1}\\|\\cdot\\|_a/C_\\infty$"],rows,"crrrrr")
ax[0].set_title("Energy error (dotted: asymptotic law (d))");ax[1].set_title("$L^2$ error");ax[2].set_title("Sampled max error")
for a in ax: a.set_xlabel("N"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/ex3.png"); plt.close()
tex_table("ex3_summary",["$s$","$p$","$q$","$2q+1$","EOC (128$\\to$256)","$C_\\infty$ (Thm (d))","ratio at $N{=}256$"],rows_rate,"cccccrr")
# quadrature sensitivity of load f_p (s=.5,p=1; log-singular forcing) and (s=.3,p=1.2)
print("EX3 quad")
R["quad"]={}; fig,ax=plt.subplots(figsize=(5.5,3.8)); Qs=[8,16,32,64,128,256,512]; rows=[]
for (s,p) in [(0.5,1.0),(0.3,1.2)]:
    N=32; q=p-s; cfull=c_closed_fast(4000,s,p); res={"GJ":[],"GL":[]}
    for Q in Qs:
        for kind in ["GJ","GL"]:
            x,w=gj(Q,s) if kind=="GJ" else np.polynomial.legendre.leggauss(Q)
            fx=fp_nodes(s,p,x); P=Pmat(N,s,x); ww=w*fx*(1 if kind=="GJ" else (1-x**2)**s)
            c=(P@ww)/(lam(np.arange(N+1),s)*hn(np.arange(N+1),s))
            res[kind].append(energy_err(c,s,cfull))
    R["quad"][f"{s},{p}"]=res
    ax.loglog(Qs,res["GJ"],"o-",label=f"Gauss-Jacobi, s={s},p={p}"); ax.loglog(Qs,res["GL"],"s--",label=f"Gauss-Legendre, s={s},p={p}")
    for i,Q in enumerate(Qs): rows.append([f"{s}",f"{p}",str(Q),sci(res['GJ'][i]),sci(res['GL'][i])])
ax.axhline(np.sqrt(np.sum((lh(np.arange(4001),.5)*c_closed_fast(4000,.5,1.0)**2)[33:])),color="gray",lw=.7)
ax.set_xlabel("Q (quadrature nodes)"); ax.set_ylabel("energy error, N=32"); ax.legend(fontsize=7); ax.set_title("Load-quadrature sensitivity")
plt.tight_layout(); plt.savefig("figs/ex3_quad.png"); plt.close()
tex_table("ex3_quad",["$s$","$p$","$Q$","Gauss--Jacobi$(s,s)$","Gauss--Legendre"],rows,"cccrr")

# ---------------- Example 4: general forcing, reference solution; conditioning ----------------
print("EX4")
fs={"f1":(lambda x:np.cos(3*x)+x*np.exp(x),400,800),"f2":(lambda x:np.abs(x),1500,3000)}
R["ex4"]={}; fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for key,(f,Nref,Qref) in fs.items():
  for s in [0.3,0.7]:
    c,_=load_coeffs(f,Nref,s,Qref); n=np.arange(Nref+1); l=lh(n,s)
    Ns4=[2,4,8,16,32,64] if key=="f2" else [2,4,8,16,32,64,128]
    E=[np.sqrt(np.sum((l*c**2)[N+1:])) for N in Ns4]
    xq,wq=np.polynomial.legendre.leggauss(1500); P=Pmat(Nref,s,xq)
    L2=[]
    for N in Ns4:
        cc=c.copy(); cc[:N+1]=0; e=(1-xq**2)**s*(cc@P); L2.append(np.sqrt(np.sum(wq*e**2)))
    eoc=[np.log(E[i]/E[i+1])/np.log(Ns4[i+1]/Ns4[i]) for i in range(len(Ns4)-1)]
    R["ex4"][f"{key},{s}"]=dict(N=Ns4,E=E,L2=L2,eoc=eoc)
    ax[0 if key=="f1" else 1].semilogy(Ns4,E,"o-",label=f"s={s}") if key=="f1" else ax[1].loglog(Ns4,E,"o-",label=f"s={s}")
    rows=[[str(N),sci(E[i]),(f"{eoc[i-1]:.2f}" if (i>0 and E[i]>1e-11) else "--"),sci(L2[i])] for i,N in enumerate(Ns4)]
    tex_table(f"ex4_{key}_s{s}".replace(".",""),["$N$","$\\|u-u_N\\|_a$","EOC","$\\|u-u_N\\|_{L^2}$"],rows,"crrr")
ax[0].set_title("$f=\\cos 3x+xe^x$ (analytic)"); ax[1].set_title("$f=|x|$ (interior kink)")
# quad nodes sensitivity for f1
rows=[]
s=0.3; cref,_=load_coeffs(fs["f1"][0],64,s,400); R["ex4q"]=[]
for Q in [4,8,12,16,24,32,48,64]:
    c,_=load_coeffs(fs["f1"][0],64,s,Q); e=np.sqrt(np.sum(lh(np.arange(65),s)*(c-cref)**2)); R["ex4q"].append((Q,e))
    rows.append([str(Q),sci(e) if e>0 else "0"])
tex_table("ex4_quad",["$Q$","$\\|u_N^{(Q)}-u_N^{(\\mathrm{ref})}\\|_a$, $N=64$"],rows,"cr")
# conditioning
rows=[]; Nc=[8,32,128,512,2048,8192]; R["cond"]={}
for s in [0.1,0.25,0.5,0.75,0.9]:
    k=[]
    for N in Nc: n=np.arange(N+1); v=lh(n,s); k.append(v.max()/v.min())
    R["cond"][s]=k; rows.append([f"{s}"]+[f"{v:.2f}" for v in k])
    ax[2].loglog(Nc,k,"o-",label=f"s={s}")
ax[2].set_title("$\\kappa(A_N)=\\max\\lambda_nh_n/\\min\\lambda_nh_n$")
for a in ax: a.set_xlabel("N"); a.legend(fontsize=7)
ax[0].set_ylabel("energy error (vs. reference $N_{ref}$)")
plt.tight_layout(); plt.savefig("figs/ex4.png"); plt.close()
tex_table("ex4_cond",["$s$"]+[f"$N{{=}}{N}$" for N in Nc],rows,"c"+"r"*len(Nc))

# ---------------- Example 5: eigenvalues ----------------
print("EX5")
def eigs(N,s,k=4):
    n=np.arange(N+1); A=np.diag(lh(n,s)); Q=N+6; x,w=gj(Q,2*s); P=Pmat(N,s,x)
    Mm=(P*w)@P.T; ev=eigh(A,Mm,eigvals_only=True); return ev[:k]
R["ex5"]={}; fig,ax=plt.subplots(1,2,figsize=(10,3.8)); lit={0.5:[1.1577738,2.7547541,4.3168010]}
for s in [0.25,0.5,0.75]:
    al=2*s; ref=eigs(200,s,6); asy=np.array([(k*np.pi/2-(2-al)*np.pi/8)**al for k in range(1,7)])
    Ns5=[4,8,16,32,64,128]; tab={N:eigs(N,s,6) for N in Ns5}
    R["ex5"][s]=dict(ref=ref.tolist(),asy=asy.tolist(),tab={N:tab[N].tolist() for N in Ns5})
    rows=[[str(k+1)]+[f"{tab[N][k]:.8f}" for N in [4,16,64]]+[f"{ref[k]:.8f}",f"{asy[k]:.6f}"]+([f"{lit[s][k]:.7f}"] if (s in lit and k<1) else ["--"]) for k in range(4)]
    tex_table(f"ex5_s{s}".replace(".",""),["$k$","$N{=}4$","$N{=}16$","$N{=}64$","$N{=}200$","asympt. (K)","literature"],rows,"crrrrrr")
    for k in range(3): ax[0].loglog(Ns5,[abs(tab[N][k]-ref[k])+1e-17 for N in Ns5],"o-",label=f"s={s}, k={k+1}")
ax[0].set_xlabel("N");ax[0].set_title("$|\\mu_k^{(N)}-\\mu_k^{(200)}|$");ax[0].legend(fontsize=6)
for s in [0.25,0.5,0.75]:
    ref=np.array(R["ex5"][s]["ref"]); asy=np.array(R["ex5"][s]["asy"]); ax[1].semilogy(range(1,7),abs(ref-asy)/ref,"o-",label=f"s={s}")
ax[1].set_title("rel. dev. from asymptotic formula"); ax[1].set_xlabel("k"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/ex5.png"); plt.close()
json.dump(R,open("results.json","w"),default=lambda o:o.tolist() if hasattr(o,"tolist") else str(o),indent=1)
print("done")
