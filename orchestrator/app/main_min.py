from fastapi import FastAPI

# Deprecated placeholder: not used. Keep to avoid import errors.
app = FastAPI(title="Deprecated main_min")

@app.get("/_removed")
async def _removed():
	return {"ok": True, "note": "Use app.main:app"}

