BOLD   := \033[1m
GREEN  := \033[32m
CYAN   := \033[36m
YELLOW := \033[33m
RESET  := \033[0m

.PHONY: limpiar detector test help docker-build docker-run

limpiar:
	clear
	clear
	clear

detector:
	@make limpiar --silent
	@echo "$(YELLOW)Ejecucion del detector de neumonia$(RESET)"
	PYTHONPATH=src uv run python -m src.detector_neumonia

test:
	@make limpiar --silent
	@echo "$(YELLOW)Ejecucion de los tests$(RESET)"
	uv run pytest

docker-build:
	docker build -t neumonia_uao .

docker-run:
	docker run --rm -it --name neumonia_uao neumonia_uao

help:
	@printf "$(BOLD)Usage: make <target>$(RESET)\n\n"
	@printf "$(GREEN)Available targets:$(RESET)\n"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "limpiar" "Clear terminal (3 times)"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "detector" "Run pneumonia detector GUI"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "test" "Run all tests"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "docker-build" "Build Docker image"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "docker-run" "Run container"
	@printf "  $(CYAN)%-14s$(RESET) %s\n" "help" "Display this help"