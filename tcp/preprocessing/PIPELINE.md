# TCP Pre-processing Pipeline for Task based analysis

# Extracting patient- and control group data from task based data
The first step for analysing our data is to segment our data such that we are only left with task based, and no resting state data.
This is done by running the `extract_subjects.py` script from our project root folder using

```python
python3 tcp/preprocessing/extract_subjects.py
```

NB! Note that this will extract all patient and control subject data, and corresponding task files solely based on the subjects _Group_. If the subject does not have data files for either _Stroop-_ or the _Hammer task_, they are still included in the separation, and should be explicitly filtered out in the next steps.

# Accounting for missing task based data