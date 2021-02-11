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
must be in the form of "year-day-hour".
For example: "2020-06-03"
[2]:In ICU time, put nan if no ICU transfer
[3]:Intubation time
[4]:death time
[5]:covid observation time
[6]:visit ID

arg1:Lab test:
numpy array

arg3:Vital sign:
numpy array
