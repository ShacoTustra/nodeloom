FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres,s3]"

ENV DATA_DIR=/data ALLOWED_HOSTS=*
VOLUME /data
EXPOSE 8188
# Listening on all interfaces inside the container; publish the port to localhost only, or put the server behind a
# proxy with authentication: there is no sign-in.
CMD ["nodeloom", "--host", "0.0.0.0", "--port", "8188"]
