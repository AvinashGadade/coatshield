console-data:
	.venv/bin/python scripts/make_console_data.py

test:
	.venv/bin/python -m pytest

.PHONY: console-data test
