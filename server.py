import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from classify import classify

app = FastAPI()

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/classify/")
async def classify_logs(request: Request, file: UploadFile = File(None)):
    # Try to get JSON body
    json_data = None
    try:
        # Only try to parse JSON if content-type is application/json
        if request.headers.get("content-type", "").startswith("application/json"):
            json_data = await request.json()
    except Exception:
        json_data = None

    has_file = file is not None and getattr(file, "filename", None) not in [None, ""]
    has_text = json_data is not None and isinstance(json_data, dict) and "log" in json_data and str(json_data.get("log", "")).strip() != ""

    # If both or neither are provided
    if (has_file and has_text) or (not has_file and not has_text):
        raise HTTPException(status_code=400, detail="Please provide either a file or a log text, not both.")

    if has_file:
        if not getattr(file, "filename", "").endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV.")
        try:
            df = pd.read_csv(file.file)
            if "source" not in df.columns or "log_message" not in df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")
            df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
            output_file = "resources/output.csv"
            df.to_csv(output_file, index=False)
            return FileResponse(output_file, media_type='text/csv')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            file.file.close()
    elif has_text:
        try:
            log_text = str(json_data["log"]) if isinstance(json_data, dict) and "log" in json_data else ""
            result = classify([("user", log_text)])
            return {"result": result[0] if result else "No result."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))