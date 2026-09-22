FROM python:3.10

ENV PYTHONUNBUFFERED=1
ENV PATH="/pipeline/.venv/bin:$PATH"

WORKDIR /pipeline

COPY ./requirements.txt ./

RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --no-cache-dir

COPY . .

ENTRYPOINT ["streamlit", "run", "quality/app/simulator_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]

