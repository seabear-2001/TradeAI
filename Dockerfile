FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云源，安装基础工具和添加deadsnakes PPA安装Python3.9
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装pip（Python3.9默认无pip）
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && rm get-pip.py

# 设定python3和pip3指向python3.9版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# 安装运行torch等库需要的系统依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 升级pip并安装Python库（分步安装，方便出错时定位）
RUN python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple pandas numpy matplotlib tqdm rich

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple torch gymnasium shimmy sb3-contrib ta

CMD ["/bin/bash"]
