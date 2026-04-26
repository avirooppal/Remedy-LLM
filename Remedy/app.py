import os

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI(title="Remedy AI")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_DIR = "./outputs/remedies-slm"

    print("Loading base model on CPU...")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        # ✅ Load tokenizer from base model, not adapter dir
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if os.path.exists(ADAPTER_DIR):
            print("Loading fine-tuned LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            model = model.merge_and_unload()
        else:
            print("Warning: Adapters not found. Using base model.")
            model = base_model

        model.eval()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
        )
        print("Model loaded successfully on CPU.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Failed to load model: {e}")


class ChatRequest(BaseModel):
    symptom: str


@app.on_event("startup")
async def startup_event():
    pass  # lazy load on first request


@app.post("/chat")
def chat(request: ChatRequest):
    load_model()

    if pipe is None:
        return {"response": "Model failed to load. Please check server logs."}

    symptom = request.symptom.strip()
    if not symptom:
        return {"response": "Please describe your health concern."}

    # ✅ Prompt format must exactly match training format
    prompt = f"""<|user|>
I have {symptom}. What home remedy and yoga pose would you suggest?

<|assistant|>
"""

    try:
        result = pipe(prompt)
        full_text = result[0]["generated_text"]

        # Extract only the assistant reply
        if "<|assistant|>" in full_text:
            response_text = full_text.split("<|assistant|>")[-1].strip()
        else:
            response_text = full_text.replace(prompt, "").strip()

        # Clean up any trailing incomplete sentence
        if response_text and not response_text[-1] in ".!?\n":
            last_stop = max(
                response_text.rfind("."),
                response_text.rfind("!"),
                response_text.rfind("?"),
            )
            if last_stop > 0:
                response_text = response_text[: last_stop + 1]

        disclaimer = "\n\n⚠️ These are traditional suggestions only. Please consult a doctor for serious or persistent conditions."
        return {"response": response_text + disclaimer}

    except Exception as e:
        return {"response": f"An error occurred: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
