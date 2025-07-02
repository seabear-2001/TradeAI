FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云源，并安装软件属性工具（添加 PPA）
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-dev build-essential curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 建立 python3 和 pip3 的软链接指向 python3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# 升级 pip 并安装 Python 包（使用阿里云 PyPI 镜像源）
RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple torch pandas numpy gymnasium matplotlib shimmy tqdm rich sb3-contrib ta

CMD ["/bin/bash"]
