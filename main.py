from fastapi import FastAPI
from model import GPT2PPL

app = FastAPI()

# Initialize the model
model = GPT2PPL()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Detect Text API!"}

@app.post("/detect-text/")
async def generate_text(sentence: str):
    try:
        results, out = model(sentence)
        return {"label": results['label'], "output": out}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
