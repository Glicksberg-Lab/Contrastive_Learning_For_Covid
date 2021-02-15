Contrastive Learning For Covid:

This Project Implement Contrastive Learning for clinical outcome prediction of Covid patient

System Requirement:
Tensorflow 1.15.0

To run the script:
python3 main.py arg1 arg2 arg3

Input data:

arg1:Registrition information:
numpy array
columns:
[0]:patient mrn id
[1]:admit time:
must be in the form of: "2020=05-20 14:52:00"
[2]:In ICU time, put nan if no ICU transfer
[3]:Intubation time
[4]:death time
[5]:covid flag: put "1" if observed, "nan" if not.

arg1:Lab test:
numpy array
columns:
[0]:patient mrn id
[1]:lab name
[2]:lab result value
[3]:lab result time:
In the form of:"20200520180800"

arg3:Vital sign:
numpy array
columns:
[0]:patient mrn id
[1]:vital sign name
[2]:vital sign result value
[3]:vital sign result time:
In the form of:"20200520180800"
The vital sign includes:
blood pressure:systolic; blood pressure:diastolic; temperature; pulse oximetry: respirations; pulse;
height; weight/scale


