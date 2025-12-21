.PHONY: cable-scan stability-map stability-plot

cable-scan:
	python scripts/run_experiment.py cable-scan

stability-map:
	python scripts/run_experiment.py stability-map

stability-plot:
	python scripts/run_experiment.py stability-plot

