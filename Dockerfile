FROM tiangolo/python-machine-learning:python3.7

LABEL maintainer="michael.graf3110@gmail.com"

COPY analysis /home/analysis
COPY requirements.txt /home/requirements.txt
EXPOSE 8888
RUN pip install -r /home/requirements.txt && pip install -e /home/analysis
CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root






