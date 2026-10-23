PY := .venv/bin/python
RUN := cd src && ../$(PY) -m

.PHONY: download process models colab-zip all

download:          ## small datasets only (~110 MB); EAGLE-I is built in Colab
	$(RUN) lightsout.download

process:           ## needs data/interim/outage_events_*.parquet (from Colab)
	$(RUN) lightsout.noaa
	$(RUN) lightsout.match
	$(RUN) lightsout.features

models:
	for nb in notebooks/0[1-4]_*.ipynb; do $(PY) -m jupyter nbconvert --to notebook --execute --inplace $$nb; done

colab-zip:         ## code bundle to upload to My Drive
	rm -f LightsOut_code.zip
	zip -qr LightsOut_code.zip src notebooks requirements.txt Makefile -x '*__pycache__*' '*.ipynb_checkpoints*'

all: download process models
