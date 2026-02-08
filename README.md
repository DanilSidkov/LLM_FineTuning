# P-Tuning v2 для Sentiment Analysis

Производственный пайплайн для эффективной настройки языковых моделей с использованием P-Tuning v2 на задаче классификации сентиментов.

## Что я использую

- **P-Tuning v2** - State-of-the-art метод parameter-efficient fine-tuning
- **Поддержка множества моделей** (BERT, GPT-2, RoBERTa, DistilBERT)
- **FastAPI REST API** для инференса
- **Docker контейнеризация**
- **WandB логирование**
- **Конфигурация через YAML**
- **Пакетный инференс**

### Структура проекта

```bash
sentiment-ptuning-v2/
├── config/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
├── experiments/
├── models/
│   └── saved_prompts/
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── train.py
├── inference_api.py
├── README.md
├── Dockerfile    
└── Makefile
```