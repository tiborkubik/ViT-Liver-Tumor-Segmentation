.PHONY: clean_cache
clean_cache:
	find src/ -type f -name "*.py[co]" -delete
	find src/ -type d -name "__pycache__" -delete

.PHONY: arch
arch:
	zip -r xpolok03_xkubik34_xondri07.zip src documentation.pdf requirements.txt run_job.sh README.md LICENSE presentation.pdf