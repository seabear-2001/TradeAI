FROM registry.aliyuncs.com/library/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple torch pandas numpy gymnasium matplotlib shimmy tqdm rich sb3-contrib ta

CMD ["/bin/bash"]
