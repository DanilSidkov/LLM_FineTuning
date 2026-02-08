.PHONY: train test api docker-build docker-run clean help

CONFIG_PATH=config/training_config.yaml
MODEL_NAME=sentiment-ptuning-v2
DOCKER_IMAGE=ptuning-sentiment
DOCKER_TAG=latest

help:
	@echo "Доступные команды:"
	@echo "  train     - Обучение модели"
	@echo "  test      - Тестирование инференса"
	@echo "  api       - Запуск API сервера"
	@echo "  docker-build - Сборка Docker образа"
	@echo "  docker-run   - Запуск Docker контейнера"
	@echo "  clean     - Очистка временных файлов"

train:
	@echo "Запуск обучения P-Tuning v2..."
	python train.py --config $(CONFIG_PATH)

train-wandb:
	@echo "Запуск обучения с WandB..."
	WANDB_API_KEY=your_key_here python train.py --config $(CONFIG_PATH)

test:
	@echo "Тестирование инференса..."
	python test_inference.py

api:
	@echo "Запуск FastAPI сервера..."
	uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	@echo "Сборка Docker образа..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	@echo "Запуск Docker контейнера..."
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

clean:
	@echo "Очистка временных файлов..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf ./experiments/*/checkpoint-*
	rm -rf ./experiments/*/runs

requirements:
	@echo "Обновление requirements..."
	pip freeze > requirements.txt

format:
	@echo "Форматирование кода..."
	black .
	isort .

lint:
	@echo "Проверка кода..."
	flake8 .
	mypy .