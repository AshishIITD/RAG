# ==============================================================================
# PHASE 6: DEVOPS - DOCKERFILE
# ==============================================================================
# We use a multi-stage build to keep the final production image small and secure.

# ---- Stage 1: Build ----
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Production ----
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the RAG application code
COPY . .

# Expose port 8501 for the Streamlit UI
EXPOSE 8501

# Command to run the UI
CMD ["streamlit", "run", "phase7_ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
