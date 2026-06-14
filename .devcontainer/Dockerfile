#Use microsoft devcontainer
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- install Rust ----
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN printf '%s\n' \
    '' \
    '# Show the current git branch in interactive Bash shells.' \
    'if [ -n "$PS1" ] && [ -f /usr/lib/git-core/git-sh-prompt ]; then' \
    '    . /usr/lib/git-core/git-sh-prompt' \
    "    PS1='\\[\\033[01;32m\\]\\u@\\h\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\[\\033[01;33m\\]\$(__git_ps1 \" (%s)\")\\[\\033[00m\\]\\$ '" \
    'fi' \
    >> /root/.bashrc

# ---- verify nvcc exists ----
RUN nvcc --version
