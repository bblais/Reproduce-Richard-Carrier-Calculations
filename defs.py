from fractions import Fraction as F
from pandas import DataFrame
import pandas as pd
pd.set_option("display.precision", 5)

from copy import deepcopy

def P2Odds(P):
    Odds=P/(1-P)
    return Odds

def Odds2P(Odds):
    P=Odds/(1+Odds)
    return P

class Evidence(object):

    def __init__(self,name):
        self.name=name
        self.labels=[]
        self.fractions=[]

    def append(self,label,fraction=None,other=None):
        if fraction is None:  # done as a string label | top/bottom
            S=label.strip()
            lines=S.split("\n")
            for line in lines:
                part1,part2=line.split("|")
                label=part1.strip()

                top,bottom=part2.split('/')
                top=int(top.strip())
                bottom=int(bottom.strip())
            
                self.append(label,top,bottom)

            return


        
        if not other is None:
            fraction=F(fraction,other)

        self.fractions.append(fraction)
        self.labels.append(label)


    @property
    def ratio(self):
        return prod(self.fractions)


    @property
    def P_h(self):
        odds=self.ratio
        P=odds/(1+odds)
        return P

    @property
    def P_neg_h(self):
        return 1-self.P_h
        

    
    def __str__(self):
        S=f"{self.name}:\n"

        mx=max([len(label) for label in self.labels])
        
        
        for l,f in zip(self.labels,self.fractions):
            l=" "*(mx-len(l))+l
            S+=f"\t{l}:\t{f}\n"

        S+="\n"
        l='Ratio'
        l=" "*(mx-len(l))+l
        S+=f"\t{l}:\t{self.ratio}={float(self.ratio)}\n"
        
        l='P(h)'
        l=" "*(mx-len(l))+l
        S+=f"\t{l}:\t{self.P_h}={float(self.P_h)}\n"
        l='P(neg h)'
        l=" "*(mx-len(l))+l
        S+=f"\t{l}:\t{1-self.P_h}={float(1-self.P_h)}\n"
        return S.strip()

    def __repr__(self):
        return str(self)

    def __mul__(self,other):

        name=' x '.join([self.name,other.name])
        E=Evidence(name)
        for l,f in zip(self.labels,self.fractions):
            E.append(self.name+" - "+l,f)
        for l,f in zip(other.labels,other.fractions):
            E.append(other.name+" - "+l,f)
            
        return E

def prod(arr):
    from numpy import prod as np_prod
    if not isinstance(arr[0],Evidence):
        return np_prod(arr)


    name=' x '.join([_.name for _ in arr])
    E=Evidence(name)
    for e in arr:
        for l,f in zip(e.labels,e.fractions):
            E.append(e.name+" - "+l,f)

    return E


class Posterior(object):

    def __init__(self,name,fraction):
        self.name=name
        self.value=fraction

        self.updates={name:fraction}

    def update(self,label,count_data_table):
        
        Nhb,Nha,Nmb,Nma=count_data_table.df.values.ravel()

        self.updates[label]=[Nhb,Nha,Nmb,Nma]
        Nh=Nhb+Nha  # total historical
        Nm=Nmb+Nma  # total not-historical
        P_h_num=F(Nha+1,Nh+2)*self.value
        P_m_num=F(Nma+1,Nm+2)*(1-self.value)

        T=P_h_num+P_m_num

        self.value=P_h_num/T


    @property
    def numerators(self):
        return self.numden[0]
    @property
    def denominator(self):
        return self.numden[1]
    
    @property
    def numden(self):
        num=[]
        
        P=self
        keys=list(reversed(P.updates.keys()))
        product=1
        T=0
        
        for key in keys:
            if key==keys[-1]: # prior
                product*=P.updates[key]
            else:
                Nhb,Nha,Nmb,Nma=P.updates[key]
                Nh=Nhb+Nha  # total historical
                Nm=Nmb+Nma  # total not-historical
                product*=F(Nha+1,Nh+2)
                
        num.append(product)
        
        T+=product

        product=1
        for key in keys:
            if key==keys[-1]: # prior
                product*=(1-P.updates[key])
            else:
                Nhb,Nha,Nmb,Nma=P.updates[key]
                Nh=Nhb+Nha  # total historical
                Nm=Nmb+Nma  # total not-historical
                product*=F(Nma+1,Nm+2)

        num.append(product)
        
        T+=product
        
        return num,T

    def __repr__(self):
        P=self
        keys=list(reversed(P.updates.keys()))
        datastr=",".join(keys)
        
        lines=[]
        
        arr=[]
        product=1
        T=0
        T_str=""
        for key in keys:
            if key==keys[-1]: # prior
                arr.append(str(P.updates[key]))
                product*=P.updates[key]
            else:
                Nhb,Nha,Nmb,Nma=P.updates[key]
                Nh=Nhb+Nha  # total historical
                Nm=Nmb+Nma  # total not-historical
                product*=F(Nha+1,Nh+2)
                
                arr.append(f"({Nha}+1)/({Nh}+2)")

        T_str=str(product)
        T+=product
        line1=f"P(h|{datastr}) ~ "+" x ".join(arr)+f"= {product} = {float(product):.5f}"
        
        arr=[]
        product=1
        for key in keys:
            if key==keys[-1]: # prior
                arr.append(str(1-P.updates[key]))
                product*=(1-P.updates[key])
            else:
                Nhb,Nha,Nmb,Nma=P.updates[key]
                Nh=Nhb+Nha  # total historical
                Nm=Nmb+Nma  # total not-historical
                product*=F(Nma+1,Nm+2)
                
                arr.append(f"({Nma}+1)/({Nm}+2)")


        T_str+=" + "+str(product)
        
        T+=product
        
        line2=f"P(neg h|{datastr}) ~ "+" x ".join(arr)+f"= {product} = {float(product):.5f}"
        
        line3=f"           T={T_str}={T} = {float(T):.5f}"
        
        P_m=product/T
        P_h=1-P_m

        line4=f"P(h|{datastr}) = {P_h} = {float(P_h):.5f} "
        line5=f"P(neg h|{datastr}) = {P_m} = {float(P_m):.5f} "

        lines=[line1,line2,line3,line4,line5]
        return "\n".join(lines)        
                

from IPython.display import display,Latex,Markdown
class Table(object):

    def __init__(self,name,Nhb,Nha,Nmb,Nma):
        self.name=name
        self.df=DataFrame({'Depicted Before 10th Century':[Nhb,Nmb],
                  'Depicted After 10th Century':[Nha,Nma]},
                 ['Historical','Non-historical'])

    def display(self):
        display(Markdown(self.markdown()))

    def markdown(self):
        df=self.df
        row1=[' ']+list(df.columns)+["Total"]
        rows=list(df.iterrows())
        vals=rows[0][1].array
        row2=['Historical',f'$N_{{hb}}={vals[0]}$',f'$N_{{ha}}={vals[1]}$',f'$N_{{h}}={sum(vals)}$']
        
        vals=rows[1][1].array
        row3=['Non-historical',f'$N_{{mb}}={vals[0]}$',f'$N_{{ma}}={vals[1]}$',f'$N_{{m}}={sum(vals)}$']
        
        vals=df.sum().array
        row4=['Total',f'$N_b={vals[0]}$',f'$N_a={vals[1]}$',f'$N={sum(vals)}$']

        S=""
        
        S+='|'
        for col in row1:
            S+=f" {col:14s}|"
        S+="\n"
        
        S+='|'
        for i,col in enumerate(row1):
            if i==0:
                S+=f"-------------:|"
            else:
                S+=f":-------------:|"
        S+="\n"
        
        
        for row in [row2,row3,row4]:
            S+='|'
            for col in row:
                S+=f" {col:14s}|"
            S+="\n"
        
        return S

def get_carrier_data(prior=None):
    data={}
    
    name='Prior Charitable'
    
    # E=Evidence(name)
    # E.append('e1,e2',F(4+1,14+2)/(1-F(4+1,14+2)))  # not rounded
    
    E=Evidence(name)

    if not prior:
        E.append('prior ($e_1$,$e_2$)',P2Odds(F(1,3))) 
    else:
        E.append('prior ($e_1$,$e_2$,$e_3$,$e_4$,$e_5$)',prior) 

    data[name]=E    

    name='Prior Uncharitable'
    E=Evidence(name)

    if not prior:
        E.append('prior ($e_1$,$e_2$)',P2Odds(F(0+1,14+2)))
    else:
        E.append('prior ($e_1$,$e_2$,$e_3$,$e_4$,$e_5$)',prior) 
    
    data[name]=E    

    name='Extrabiblical Charitable'
    E=Evidence(name)
    E.append("""
    Twin traditions | 4/5
    Documentary silence | 1/1
    1 Clement | 4/5
    Ignatius and Ascension of Isaiah | 4/5
    Papias | 1/1
    Hegesippus | 9/10
    Josephus | 1/1
    Pliny  | 1/1
    Tacitus | 1/1
    Suetonius | 1/1
    Thallus | 1/1
    Lack of gainsaying witnesses | 1/1
    """)
    
    data[name]=E
    
    name='Extrabiblical Uncharitable'
    E=Evidence(name)
    E.append("""
    Twin traditions | 1/2
    Documentary silence | 1/1
    1 Clement | 1/2
    Ignatius and Ascension of Isaiah | 1/2
    Papias | 1/1
    Hegesippus | 4/5
    Josephus | 1/1
    Pliny  | 1/1
    Tacitus | 1/1
    Suetonius | 1/1
    Thallus | 1/1
    Lack of gainsaying witnesses | 1/1
    """)
    
    
    data[name]=E
    
    
    name='Acts Charitable'
    
    E=Evidence(name)
    E.append('Vanishing family et al.',4,5)
    E.append("Omissions in Paul's trials",9,10)
    E.append("Remainder of Acts",1,1)
    
    data[name]=E
    
    name='Acts Uncharitable'
    
    E=Evidence(name)
    E.append('Vanishing family et al.',2,5)
    E.append("Omissions in Paul's trials",1,2)
    E.append("Remainder of Acts",1,1)
    
    data[name]=E
    
    name='Gospels Charitable'
    E=Evidence(name)
    E.append("""
    Reasons | 1/1
    """)
    
    
    data[name]=E
    
    name='Gospels Uncharitable'
    E=Evidence(name)
    E.append("""
    Reasons | 1/1
    """)
    
    
    data[name]=E
    
    
    name='Epistles Charitable'
    E=Evidence(name)
    E.append("""
    Other canonical Epistles | 4/5
    Gospels in Paul, Hebrews, Colossians | 3/5
    Things Jesus said | 1/1
    The Eucharist (1 Cor. 11.23-26) | 1/1
    Things Jesus did | 3/4
    Made from sperm | 2/1
    Made from a woman | 2/1
    Brothers of the Lord | 2/1
    """)
    
    
    data[name]=E
    
    name='Epistles Uncharitable'
    E=Evidence(name)
    E.append("""
    Other canonical Epistles | 3/5
    Gospels in Paul, Hebrews, Colossians | 2/5
    Things Jesus said | 1/1
    The Eucharist (1 Cor. 11.23-26) | 1/1
    Things Jesus did | 1/2
    Made from sperm | 1/1
    Made from a woman | 1/1
    Brothers of the Lord | 1/2
    """)
    
    
    data[name]=E
    
    
    
    charitable=deepcopy([data[key] for key in data if 'Charitable' in key])
    for T in charitable:
        T.name=T.name.replace(' Uncharitable','').replace(' Charitable','')
    
    uncharitable=deepcopy([data[key] for key in data if 'Uncharitable' in key])
    for T in charitable:
        T.name=T.name.replace(' Uncharitable','').replace(' Charitable','')
    
    Tc=prod(charitable)
    Tu=prod(uncharitable)
    
    labels_c=[s.split(' - ')[1] for s in Tc.labels]
    labels_u=[s.split(' - ')[1] for s in Tu.labels]
    
    assert all([_1==_2 for _1,_2 in zip(labels_c,labels_u)])

    labels=[]
    for i,label in enumerate(labels_c):
        labels.append(f'$c_{{{i+1}}}'+" :=$ "+f"{label}")
    
    df=DataFrame({'a fortiori':[str(f) for f in Tc.fractions],
                  'a judicantiori':[str(f) for f in Tu.fractions],
                 'Source':[s.split(' - ')[0] for s in Tc.labels]},
                 labels)    

    return df,Tc,Tu

class TableCarrier(object):

    def __init__(self,name,Nhc,Nmc,Nhu,Nmu):
        self.name=name
        self.df=DataFrame({'a fortiori':[Nhc,Nmc],
                  'a judicantiori':[Nhu,Nmu]},
                 ['Historical','Non-historical'])

    def display(self):
        display(Markdown(self.markdown()))

    def markdown(self):
        df=self.df
        row1=[' ']+list(df.columns)
        rows=list(df.iterrows())
        vals=rows[0][1].array
        row2=['Historical',f'$N_h={vals[0]}$',f'$N_h={vals[1]}$']
        
        vals=rows[1][1].array
        row3=['Non-historical',f'$N_m={vals[0]}$',f'$N_m={vals[1]}$']
        
        vals=df.sum().array
        row4=['Total',f'$N={vals[0]}$',f'$N={vals[1]}$']


        S='|'
        for col in row1:
            S+=f" {col:14s}|"
        S+="\n"
        
        S+='|'
        for i,col in enumerate(row1):
            if i==0:
                S+=f"-------------:|"
            else:
                S+=f":-------------:|"
        S+="\n"
        
        
        for row in [row2,row3,row4]:
            S+='|'
            for col in row:
                S+=f" {col:14s}|"
            S+="\n"
        
        return S


class LS(object):
    def __init__(self,h,N):
        self.h=h
        self.N=N

    @property
    def value(self):
        return F(self.h+1,self.N+2)

    def __float__(self):
        return float(self.value)
        
    def __str__(self):
        S=r"\left(\frac{%d+1}{%d+2}\right)" % (self.h,self.N)
        return S        

    def __mul__(self,other):
        try:
            return self.value*other
        except TypeError:
            return self.value*other.value

    __rmul__ = __mul__
    
    def __repr__(self):
        return f"Fraction({self.h}+1,{self.N}+2)"


def extra_to_display(extra,term):
    S1=f"P({term})"

    if isinstance(extra,str):
        S2=r"\underbrace{"+S1+"}_{"+ extra  +"}"
    else:  # rule of succession
        S2=r"\underbrace{"+S1+r"}_{\left(\frac{%s+1}{%s+2}\right)}" % (extra[0],extra[1])

    return S2

def extra_to_display2(extra):
    if isinstance(extra,str):
        S2=extra
    else:  # rule of succession
        S2=r"\left(\frac{%s+1}{%s+2}\right)" % (extra[0],extra[1])

    return S2
    
def P_to_display(PP):
    S=""
    
    S=r"\begin{aligned}"+"\n"
    for key in PP:
        S+=f"P({key})&\\sim "

        terms=[]
        extras=[]
        for _ in PP[key]:
            terms.append(_[0])
            if len(_)==3:
                extras.append(_[2])
        if not extras:        
            S+=" \\cdot ".join([f"P({term})" for term in terms])
        else:
            S+=" \\cdot ".join([f"{extra_to_display(extra,term)}" for term,extra in zip(terms,extras)])
            
        S+=r"\\"+"\n"
    
    S+=r"\end{aligned}"+"\n"
    
    display(Markdown(S))
    
    S=""
    S=r"\begin{aligned}"+"\n"
    numerators=[]
    for key in PP:
        S+=f"P({key})&\\sim "
        vals=[_[1] for _ in PP[key]]
        S+=" \\cdot ".join([f"{val}" for val in vals])
        S+=f" = {float(prod(vals)):.4f}"
        numerators.append(float(prod(vals)))
        S+=r"\\"+"\n"
    S+="T_{\\text{denominator}}&="+" + ".join([f"{num:.4f}" for num in numerators])+f" = {sum(numerators):.4f}"
    S+=r"\end{aligned}"+"\n"
    T=sum(numerators)
    display(Markdown(S))
    
    S=""
    S=r"\begin{aligned}"+"\n"
    probs=[]
    numerators=[]
    for key in PP:
        S+=f"P({key})& "
        vals=[_[1] for _ in PP[key]]
        S+=f"={float(prod(vals)):.4f}/{T:.4f}"
        S+=f"={float(prod(vals)/T):.4f}"
        probs.append(prod(vals)/T)
        S+=r"\\"+"\n"
    S+=r"\end{aligned}"+"\n"
    
    display(Markdown(S))

    return probs


def P_to_display2(PP):
    S=""

    S+="1. "
    S+=r"$$\begin{aligned}"+"\n"
    for key in PP:
        S+=f"P({key})&\\sim "

        terms=[]
        extras=[]
        for _ in PP[key]:
            terms.append(_[0])
            if len(_)==3:
                extras.append(_[2])
        if not extras:        
            S+=" \\cdot ".join([f"P({term})" for term in terms])
        else:
            S+=" \\cdot ".join([f"{extra_to_display(extra,term)}" for term,extra in zip(terms,extras)])
            
        S+=r"\\"+"\n"
    
    S+=r"\end{aligned}$$"+"\n"
    
    S+=r"$$\begin{aligned}"+"\n"
    numerators=[]
    
    for key in PP:
        
        extras=[]
        vals=[]
        for _ in PP[key]:
            vals.append(_[1])
            if len(_)==3:
                extras.append(_[2])
        
        S+=f"P({key})&\\sim "

        if extras:
            S+=" \\cdot ".join([f"{extra_to_display2(extra)}" for extra in extras])
            S+=r"\\"+"\n & \\sim "
        else:        
            pass
        S+=" \\cdot ".join([f"{val}" for val in vals])
        S+=f" = {float(prod(vals)):.4f}"
        numerators.append(float(prod(vals)))
        S+=r"\\"+"\n"
    S+=r"\end{aligned}$$"+"\n"

    S+="2. "
    S+=r"$$\begin{aligned}"+"\n"
    S+="T_{\\text{denominator}}&="+" + ".join([f"{num:.4f}" for num in numerators])+f" = {sum(numerators):.4f}"
    S+=r"\end{aligned}$$"+"\n"
    T=sum(numerators)

    
    S+="3. "
    S+=r"$$\begin{aligned}"+"\n"
    probs=[]
    numerators=[]
    for key in PP:
        S+=f"P({key})& "
        vals=[_[1] for _ in PP[key]]
        S+=f"={float(prod(vals)):.4f}/{T:.4f}"
        S+=f"={float(prod(vals)/T):.4f}"
        probs.append(prod(vals)/T)
        S+=r"\\"+"\n"
    S+=r"\end{aligned}$$"+"\n"
    
    display(Markdown(S))

    return probs,S


        