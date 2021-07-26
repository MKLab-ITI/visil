FROM python:3.5

WORKDIR /

RUN apt-get update && apt install -y libgl1-mesa-glx

ADD ./ ./

RUN wget http://ndd.iti.gr/visil/ckpt.zip

RUN unzip ckpt.zip

RUN python -m pip install pip --upgrade

RUN python -m pip install numpy --upgrade

RUN python -m pip uninstall tensorboard-data-server

RUN python -m pip install tqdm>=4.2

RUN python -m pip install tensorflow==1.15.4

RUN python -m pip install opencv-python>=3.1.0
