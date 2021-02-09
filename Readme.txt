Contrastive Learning For Covid:

Data Input requires:
1.Lab test: 
mxn numpy array, m:dimension for all available lab test data

2.Registrition information:
mxn numpy array
rows:dimension for all registry data.
columns:

3.Vital sign:
mxn numpy array

4.Covid patient:
mxn numpy array, if patient has multiple visit, pick the last visit data

columns(from 0 to the end):
[0]:patient mrn id
[1]:patient admit time
[2]:


