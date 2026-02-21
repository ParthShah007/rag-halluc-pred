.PHONY: install data train app api test docker clean help

help:  ## Show this help
	@echo "RAG-Optimize - Available commands:"
	@echo "  make install   Install Python dependencies"
	@echo "  make data      Load and preprocess data"
	@echo "  make train     Train all models"
	@echo "  make app       Run Streamlit dashboard"
	@echo "  make api       Run FastAPI server"
	@echo "  make test      Run unit tests"
	@echo "  make docker    Build and run Docker container"
	@echo "  make clean     Remove generated files"

install:  ## Install dependencies
	pip install -r requirements.txt

data:  ## Load and merge datasets
	python -c "from src.data.load_data import load_and_merge, save_processed; save_processed(load_and_merge())"

train:  ## Train all models and save artifacts
	python -m src.models.train

app:  ## Launch Streamlit dashboard
	streamlit run app/streamlit_app.py --server.port 8501

api:  ## Launch FastAPI server
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

test:  ## Run unit tests
	python -m pytest tests/ -v

docker:  ## Build and run Docker image
	docker build -t rag-optimizer .
	docker run -p 8501:8501 -p 8000:8000 rag-optimizer

clean:  ## Remove generated files
	rm -rf models/*.joblib models/*.json
	rm -rf data/processed/*
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	rm -rf catboost_info/
