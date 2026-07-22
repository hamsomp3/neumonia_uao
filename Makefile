.PHONY: limpiar detector test


limpiar:
	rm -f *.o
	rm -f *.out
	rm -f *.exe
	rm -f *.log
	clear
	clear
	clear

detector:
	make limpiar
	@echo "Ejecucion del detector de neumonia"
	PYTHONPATH=src uv run python -m src.detector_neumonia

test:
	make limpiar
	@echo "Ejecucion de los tests"
	uv run pytest