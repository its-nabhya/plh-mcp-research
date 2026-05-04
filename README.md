# Protocol-Level Hallucination (PLH) Detection in MCP Agents

## Overview
Research project on detecting hallucinations in tool-augmented LLM agents using MCP Worlds.

## Structure
- worlds: MCP environments
- agent: tool-using agent
- logging: validation + logs
- labeling: hallucination taxonomy
- models: ML detectors

## Run
python scripts/run_pipeline.py


## How to run (Windows based device, please check the equivalent prompts fo Mac and linux):
### Create a Virtual environment
```
python -m venv .venv
.venv\Scripts\activate
```
Make sure that `.gitignore` filehas `.vemv/` path

### install all dependencies
``` 
python.exe -m pip install --upgrade pip
pip install anthropic pyyaml numpy scikit-learn pytest requests
pip install -r requirements.txt
```
Note: Ensure you have Ollama Locally installed and use a separate terminal to run the local server with the model as per your requirement

### Collect data from MCP Worlds by running prompts
```
python scripts/run_pipeline.py --mode live --provider ollama --model qwen2.5:7b --num 31 --sleep 1.0 --run-name ollama_run_1
```
Or 
modify the world_prompt_bank.py ---> modify the scripts/run_all_worlds_and_build_dataset.sh file and generate the data

### Rebuild splits from your existing logs
```
python -c "from src.pipeline.dataset_builder import DatasetBuilder; db = DatasetBuilder('data/logs'); entries = db.filter(db.load_all()); train, val, test = db.split(entries); db.save_splits(train, val, test, 'data/features')"
```

### Train
```
python scripts/train_model.py --data data/features --model lr
```
(lr here represents logistic regression passed as a parameter to the script being executed) 

### Evaluate
```
python scripts/evaluate_model.py --model data/models/lr_binary.pkl --data data/features 
```