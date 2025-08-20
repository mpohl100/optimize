#Use microsoft devcontainer
ARG VARIANT=bullseye
FROM mcr.microsoft.com/vscode/devcontainers/rust:1-${VARIANT}

# Install cargo-tarpaulin
RUN cargo install cargo-tarpaulin

# Install Node.js (empfohlen: NodeSource)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs
