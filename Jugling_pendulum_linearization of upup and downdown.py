#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sympy as sp


# In[3]:


m1, m2, l1, l2, g, b1, b2 = sp.symbols("m_1, m_2, l_1, l_2, g, b_1, b_2")
t = sp.Symbol("t")

phi1 = sp.Function("\phi_1")
phi2 = sp.Function("\phi_2")
u = sp.Function("u")


# In[4]:


M = sp.Matrix([
    [(m1 + m2)*l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2,
     -m2 * l2**2 - m2 * l1 * l2],
    [-m2 * l2**2 - m2 * l1 * l2,
    m2 * l2**2]
])
M


# In[5]:


G = sp.Matrix([
    [b1, 0],
    [0, b2]
])
G


# In[6]:


K = sp.Matrix([
    [-(m1 + m2) * g * l1 - m2 * g * l2,
     m2 * g * l2], 
    [m2 * g * l2, -m2 * g * l2]
])
K


# In[7]:


phi = sp.Matrix([
    phi1(t),
    phi2(t)
])

phi_dot = sp.Matrix([
    sp.diff(phi1(t), t),
    sp.diff(phi2(t), t)
])


# In[8]:


phi_double_dot = M.inv() * (sp.Matrix([u(t), 0]) - (G * phi_dot) - (K * phi))
phi_double_dot


# In[9]:


x1 = sp.Function(r"x_1")
x2 = sp.Function(r"x_2")
x3 = sp.Function(r"x_3")
x4 = sp.Function(r"x_4")

substitution = {phi1(t): x1(t), phi2(t): x2(t), sp.diff(phi1(t), t): x3(t), sp.diff(phi2(t), t) : x4(t)}

f = sp.Matrix([
    x3(t),
    x4(t),
    phi_double_dot[0].subs(substitution),
    phi_double_dot[1].subs(substitution)
])

f


# In[10]:


physical_model = {
    m1: 0.125,
    m2: 0.05,
    l1: 0.075,
    l2: 0.15,
    b1: 0.0048,
    b2: 0.00002,
    g: 9.81
}

f_no_control = f.subs({u(t) : 0})
f_no_control_phys = f_no_control.subs(physical_model)
f_no_control_phys


# In[11]:


up_up = {x1(t): 0, x2(t): 0, x3(t): 0, x4(t): 0}
A_up_up = f_no_control_phys.jacobian(sp.Matrix([x1(t), x2(t), x3(t), x4(t)])).subs(up_up)
A_up_up


# In[12]:


A_up_up.eigenvals()


# In[ ]:





# In[25]:


K_2 = sp.Matrix([
    [(m1 + m2) * g * l1 + m2 * g * l2,
     -m2 * g * l2], 
    [-m2 * g * l2, m2 * g * l2]
])
K_2


# In[26]:


phi_double_dot_2 = M.inv() * (sp.Matrix([u(t), 0]) - (G * phi_dot) - (K_2 * phi))
phi_double_dot_2


# In[30]:


x1 = sp.Function(r"x_1")
x2 = sp.Function(r"x_2")
x3 = sp.Function(r"x_3")
x4 = sp.Function(r"x_4")

substitution = {phi1(t): x1(t), phi2(t): x2(t), sp.diff(phi1(t), t): x3(t), sp.diff(phi2(t), t) : x4(t)}

f = sp.Matrix([
    x3(t),
    x4(t),
    phi_double_dot_2[0].subs(substitution),
    phi_double_dot_2[1].subs(substitution)
])

f


# In[31]:


physical_model = {
    m1: 0.125,
    m2: 0.05,
    l1: 0.075,
    l2: 0.15,
    b1: 0.0048,
    b2: 0.00002,
    g: 9.81
}

f_no_control = f.subs({u(t) : 0})
f_no_control_phys = f_no_control.subs(physical_model)
f_no_control_phys


# In[32]:


down_down = {x1(t): sp.pi, x2(t): 0, x3(t): 0, x4(t): 0}
A_down_down = f_no_control_phys.jacobian(sp.Matrix([x1(t), x2(t), x3(t), x4(t)])).subs(down_down)
A_down_down


# In[ ]:


# Values for the down down are almost the same only differs by sign for some elements. 
# Don't know if it is correct
# Eigenvalues have negative real part and an imaginary part


# In[33]:


A_down_down.eigenvals()


# In[ ]:




