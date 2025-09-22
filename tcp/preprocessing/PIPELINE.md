# TCP Pre-processing Pipeline for Task based analysis

# Extracting patient- and control group data from task based data
The first step for analysing our data is to segment our data such that we are only left with task based, and no resting state data.
This is done by running the `extract_subjects.py` script from our project root folder using

```python3
python3 tcp/preprocessing/extract_subjects.py
```

NB! Note that this will extract all patient and control subject data, and corresponding task files solely based on the subjects _Group_. If the subject does not have data files for either _Stroop-_ or the _Hammer task_, they are still included in the separation, and should be explicitly filtered out in the next steps.

# Accounting for missing task based data
We would like to only process data about subjects that have actually had task based scans. This means that we have to filter the data, which we can to directly on the output from the extracted data from the previous step, in order to not separate data twice. This initial filtering is handled by `filter_subjects.py` with utilities for creating a filtering pipeline and abstract filter classes, but is written to utilise _dependency injection_, and can therefore be used later on in the pipeline, or easily extended to for further filtering.

```python3
python3 tcp/preprocessing/filter_subjects.py
```

# Fetching actual MRI data with Datalad
As anatomical and fMRI scans result in large data files, these are not shipped in their entirety with the dataset, by default. We need to fetch the actual content of these data files using Datalad and Git Annex. Seeing as the dataset's true size boarder on 1TB in storage space, we are only interested in installing the specific data we want, and omit all other data. This means we selectively fetch task-based scan data, while we skip resting state data in its entirety. To do this, the script `fetch_filtered_data.py` iterates over included subjects computed by the previous step in the pipeline, and further uses file path pattern matching to call `datalad get` on all relevant files.

You can run this script with a dry run for testing with

```python3
python3 tcp/preprocessing/fetch_filtered_data.py --dry-run
```

When installing, you can choose to use the default selection of data files – currently set to include everything for task-relevant scans – with

```python3
python tcp/preprocessing/fetch_filtered_data.py
```

Alternatively, you can fetch specific data types:

```python3
python tcp/preprocessing/fetch_filtered_data.py --data-types raw_nifti_hammer events_hammer anatomical_t1w
```
