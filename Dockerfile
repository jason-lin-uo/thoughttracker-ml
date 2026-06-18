FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.org/simple/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN test -f models/stance-classifier/config.json \
    && test -f models/stance-classifier/model.safetensors \
    && test -f models/topic-relevance-classifier-supervalidation-hardneg2x-l512/config.json \
    && test -f models/topic-relevance-classifier-supervalidation-hardneg2x-l512/model.safetensors \
    && test -f models/topic-reranker-tfidf-sgd-supervalidation/topic_reranker_model.pkl \
    && for artifact in \
      models/stance-classifier/model.safetensors \
      models/topic-relevance-classifier-supervalidation-hardneg2x-l512/model.safetensors \
      models/topic-reranker-tfidf-sgd-supervalidation/topic_reranker_model.pkl; do \
      if head -c 128 "$artifact" | grep -q "version https://git-lfs.github.com/spec/v1"; then \
        echo "Runtime model artifact is still a Git LFS pointer: $artifact"; \
        exit 1; \
      fi; \
    done

# Runtime requires committed/restored model artifacts; missing models fail clearly.
ENV TOPIC_RELEVANCE_MAX_LENGTH=512 \
    API_HOST=0.0.0.0 \
    API_PORT=7860

EXPOSE 7860

# Hugging Face Docker Spaces require port 7860. Local/docker users can still
# override API_PORT if they want to bind the ML service on a different port.
CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-7860}"]
