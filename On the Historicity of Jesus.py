#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from defs import *


# In[ ]:


data_e12=TableCarrier("Carrier's prior",4,10,0,14)
data_e12.display()


# In[ ]:


print(data_e12.markdown())


# In[ ]:


# charitable
N_h,_,N_m,_=data_e12.df.values.ravel()
N=N_h+N_m


# In[ ]:


PP={'h|e_1,e_2':[ ['e_2|h,e_1',LS(N_h,N)],
                  ['h|e_1',F(1,2)]],
    '\\neg h|e_1,e_2':[ 
            ['e_2|\\neg h,e_1',LS(N_m,N)],
            ['\\neg h|e_1',F(1,2)]
       ],
      }

vals,S=P_to_display2(PP)


# In[ ]:


PP={'h|e_1,e_2':[ ['e_2|h,e_1',LS(N_h,N),['N_h','N']],
                  ['h|e_1',F(1,2),'1/2']],
    '\\neg h|e_1,e_2':[ 
            ['e_2|\\neg h,e_1',LS(N_m,N),['N_m','N']],
            ['\\neg h|e_1',F(1,2),'1/2']
       ],
      }

vals,S=P_to_display2(PP)


# In[ ]:


print(S)


# In[ ]:


df,Tc,Tu=get_carrier_data()


# In[ ]:


display(Markdown("#### Likeihood ratios for each piece of evidence, $c_i$"))
display(Markdown(df.to_markdown()))


# In[ ]:


display(Markdown(r"""## Charitable calculation,

- Posterior ratio $$
\frac{P(h|e_1,e_2,\left\{c_i\right\})}{P(\neg h|e_1,e_2,\left\{c_i\right\})}=\frac{%d}{%d}
$$
- Posterior probabilities $$\begin{aligned}
P(h|e_1,e_2,\left\{c_i\right\})&=\frac{%d}{%d}=%.3f\\
P(\neg h|e_1,e_2,\left\{c_i\right\})&=\frac{%d}{%d}=%.3f\\
\end{aligned}
$$

""" % (Tc.ratio.numerator,Tc.ratio.denominator,
       Tc.P_h.numerator,Tc.P_h.denominator,float(Tc.P_h),
       Tc.P_neg_h.numerator,Tc.P_neg_h.denominator,float(Tc.P_neg_h),
      )
))


display(Markdown(r"""
### Uncharitable calculation,

- Posterior ratio $$
\frac{P(h|e_1,e_2,\left\{c_i\right\})}{P(\neg h|e_1,e_2,\left\{c_i\right\})}=\frac{%d}{%d}
$$
- Posterior probabilities $$\begin{aligned}
P(h|e_1,e_2,\left\{c_i\right\})&=\frac{%d}{%d}=%f\\
P(\neg h|e_1,e_2,\left\{c_i\right\})&=\frac{%d}{%d}=%f\\
\end{aligned}
$$

""" % (Tu.ratio.numerator,Tu.ratio.denominator,
       Tu.P_h.numerator,Tu.P_h.denominator,float(Tu.P_h),
       Tu.P_neg_h.numerator,Tu.P_neg_h.denominator,float(Tu.P_neg_h),
      )
))


# In[ ]:




