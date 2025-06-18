FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run backend in background and frontend (Streamlit) in foreground
CMD ["bash", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]



