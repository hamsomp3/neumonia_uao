.PHONY: limpiar stream back front detector test


limpiar:
	rm -f *.o
	rm -f *.out
	rm -f *.exe
	rm -f *.log
	clear
	clear
	clear

stream:
	make limpiar
	@echo "Ejecucion del streamlit"
	uv run streamlit run prueba.py
	
back:
	make limpiar
	@echo "Ejecucion del backend"
	uv run uvicorn main:app --reload

front:
	make limpiar
	@echo "Ejecucion del frontend"
	uv run uvicorn main:app --reload

detector:
	make limpiar
	@echo "Ejecucion del detector de neumonia"
	uv run detector_neumonia.py

test:
	make limpiar
	@echo "Ejecucion de los tests"
	uv run pytest