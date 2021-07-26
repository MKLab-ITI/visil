FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /

RUN apt-get update --fix-missing && \
    apt-get clean && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget

RUN git clone --depth 1 https://github.com/MKLab-ITI/visil /visil

RUN wget http://ndd.iti.gr/visil/ckpt.zip

RUN unzip ckpt.zip -d /visil

RUN python -m pip install --upgrade pip 

RUN python -m pip install --upgrade numpy tqdm>=4.2 opencv-python>=3.1.0
