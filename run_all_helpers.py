import numpy as np
def sci(x):
    if x==0: return "0"
    e=int(np.floor(np.log10(abs(x)))); m=x/10**e; return f"${m:.2f}\\times10^{{{e}}}$"
def tex_table(fn,header,rows,colfmt):
    with open(f"tables/{fn}.tex","w") as f:
        f.write("\\begin{tabular}{"+colfmt+"}\n\\toprule\n"+" & ".join(header)+" \\\\\n\\midrule\n")
        for r in rows: f.write(" & ".join(r)+" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
