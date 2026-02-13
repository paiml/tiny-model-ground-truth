FROM rust:1.84-bookworm AS builder

RUN cargo install apr-cli ruchy --locked

FROM python:3.11-slim-bookworm

COPY --from=builder /usr/local/cargo/bin/apr /usr/local/bin/apr
COPY --from=builder /usr/local/cargo/bin/ruchy /usr/local/bin/ruchy

RUN pip install uv

RUN useradd -m -s /bin/bash tmgt
USER tmgt

WORKDIR /home/tmgt/app
COPY --chown=tmgt:tmgt . .

RUN uv sync

CMD ["make", "ci"]
