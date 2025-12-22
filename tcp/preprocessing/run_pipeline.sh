conda activate masters_thesis

python3 tcp/preprocessing/run_pipeline.py \
--data-source-type combined \
--hcp-root /cluster/projects/itea_lille-ie/Transdiagnostic/output/hcp_output \
--hcp-parcellated-output /Volumes/Seagate/specialisation-project/Data/hcp_output \
--duplicate-resolution prefer_hcp \
--ignore-completed \
--analysis-group primary \
--data-type timeseries
