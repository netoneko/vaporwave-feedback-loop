FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
    python \
    ipython \
    python-numpy \
    python-scipy \
    python-sklearn \
    python-skimage && \
    apt-get -y autoremove && \
    apt-get autoclean

ADD requirements.txt /opt/vaporwave/requirements.txt

WORKDIR /opt/vaporwave

#RUN pip install -r requirements.txt

ADD . /opt/vaporwave

RUN echo 'export PS1="「ＶＡＰＯＲＷＡＶＥ フィードバックループ」\w ー "' >> /root/.bashrc

CMD python predict.py
