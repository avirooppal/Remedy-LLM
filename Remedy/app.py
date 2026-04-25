import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

app = FastAPI(title="🌿 Remedy AI - Minimalist Chatbot")

# Mount static files for the UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Lazy-load model
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return
        
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_DIR = "./outputs/remedies-slm"
    
    print("Loading base model for inference...")
    
    try:
        # Load base model with 4-bit precision for extremely low resource usage
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map="auto",
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if os.path.exists(ADAPTER_DIR):
            print("Loading fine-tuned LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        else:
            print("Warning: Fine-tuned adapters not found. Using base model.")
            model = base_model
            
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

class ChatRequest(BaseModel):
    symptom: str

@app.on_event("startup")
async def startup_event():
    # Model loaded lazily upon first request to speed up startup, or can load here.
    # We will load on first request to keep it minimal and fast for testing UI.
    pass

@app.post("/chat")
def chat(request: ChatRequest):
    load_model()
    if pipe is None:
        return {"response": "Model is not loaded. Please check server logs."}
        
    # Construct prompt in the format we trained on
    prompt = f"### Instruction:\nSuggest a home remedy for {request.symptom}.\n\n### Input:\n\n### Response:\n"
    
    try:
        result = pipe(prompt)
        full_text = result[0]['generated_text']
        
        # Extract everything after "### Response:\n"
        if "### Response:\n" in full_text:
            response_text = full_text.split("### Response:\n")[-1].strip()
        else:
            response_text = full_text.replace(prompt, "").strip()
            
        disclaimer = "\n\n⚠️ This is a traditional home remedy suggestion. Please consult a doctor for serious or persistent conditions."
        return {"response": response_text + disclaimer}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    # run with `uvicorn app:app --reload`
    uvicorn.run("app:app", host="127.0.0.0", port=8000, reload=True)
