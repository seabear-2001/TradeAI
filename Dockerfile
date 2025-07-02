FROM registry.cn-hangzhou.aliyuncs.com/luban-hub/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 阿里云源加速 apt
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 使用阿里云镜像源加速
RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple \
        torch pandas numpy gymnasium matplotlib shimmy tqdm rich sb3-contrib ta

CMD ["/bin/bash"]
