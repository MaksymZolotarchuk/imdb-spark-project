FROM spark:python3

WORKDIR /app

COPY requirements.txt /app/requirements.txt

USER root
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py"]