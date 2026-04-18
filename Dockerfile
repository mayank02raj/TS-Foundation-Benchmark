FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY ts_benchmark/ ts_benchmark/
RUN pip install --no-cache-dir -e .
RUN mkdir -p /app/output

VOLUME ["/app/output"]
ENTRYPOINT ["python", "-m", "ts_benchmark"]
CMD ["--help"]
