from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel, PeftConfig
import yaml
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="P-Tuning v2 Sentiment Analysis API",
    description="API для классификации сентиментов с использованием P-Tuning v2",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    text: str
    return_probs: bool = False
    threshold: float = 0.5

class BatchSentimentRequest(BaseModel):
    texts: List[str]
    return_probs: bool = False

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int
    probabilities: Dict[str, float] = None

class ModelManager:
    """Менеджер для загрузки и управления моделью"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self.load_config(config_path)
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_id: str = "default") -> None:
        """Загрузка P-Tuning v2 модели"""
        try:
            model_config = self.config['models'][model_id]
            
            peft_config = PeftConfig.from_pretrained(model_config['path'])
            
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=model_config['num_labels']
            )
            
            model = PeftModel.from_pretrained(base_model, model_config['path'])
            model = model.to(self.device)
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
            
            self.models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            
            logger.info(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def predict_single(self, text: str, model_id: str = "default", 
                      return_probs: bool = False) -> Dict[str, Any]:
        """Предсказание для одного текста"""
        if model_id not in self.models:
            self.load_model(model_id)
        
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config['inference']['max_length'],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = F.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)
            
            label_map = self.config['models'][model_id].get('label_map', 
                                                          {0: "negative", 1: "positive"})
            
            result = {
                "text": text,
                "sentiment": label_map[predicted_class.item()],
                "confidence": confidence.item(),
                "label": predicted_class.item()
            }
            
            if return_probs:
                result["probabilities"] = {
                    label_map[i]: probs[0][i].item() 
                    for i in range(len(label_map))
                }
        
        return result
    
    def predict_batch(self, texts: List[str], model_id: str = "default",
                     return_probs: bool = False) -> List[Dict[str, Any]]:
        """Пакетное предсказание"""
        if model_id not in self.models:
            self.load_model(model_id)
        
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config['inference']['max_length'],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            confidences, predicted_classes = torch.max(probs, dim=-1)
        
        label_map = self.config['models'][model_id].get('label_map', 
                                                      {0: "negative", 1: "positive"})
        
        results = []
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "sentiment": label_map[predicted_classes[i].item()],
                "confidence": confidences[i].item(),
                "label": predicted_classes[i].item()
            }
            
            if return_probs:
                result["probabilities"] = {
                    label_map[j]: probs[i][j].item() 
                    for j in range(len(label_map))
                }
            
            results.append(result)
        
        return results

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при запуске сервиса"""
    logger.info("Starting up Sentiment Analysis API...")
    for model_id in model_manager.config['models'].keys():
        model_manager.load_model(model_id)
    logger.info("All models loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(model_manager.models.keys())
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Эндпоинт для предсказания сентимента"""
    try:
        result = model_manager.predict_single(
            text=request.text,
            return_probs=request.return_probs
        )
        
        if request.threshold and result["confidence"] < request.threshold:
            result["sentiment"] = "neutral"
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[SentimentResponse])
async def predict_batch_sentiment(request: BatchSentimentRequest):
    """Эндпоинт для пакетного предсказания"""
    try:
        results = model_manager.predict_batch(
            texts=request.texts,
            return_probs=request.return_probs
        )
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info/{model_id}")
async def get_model_info(model_id: str):
    """Информация о загруженной модели"""
    if model_id not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = model_manager.models[model_id]
    config = model_manager.config['models'][model_id]
    
    info = {
        "model_id": model_id,
        "base_model": config.get('base_model', 'unknown'),
        "num_labels": config.get('num_labels', 2),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(model_manager.device)
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)