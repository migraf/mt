FROM tiangolo/python-machine-learning:python3.7

LABEL maintainer="michael.graf3110@gmail.com"

COPY analysis /opt/analysis
COPY requirements.txt /opt/requirements.txt
COPY example_notebook.ipynb /home/analysis
WORKDIR /home/analysis
EXPOSE 8888
RUN pip install -r /opt/requirements.txt && pip install -e /opt/analysis
CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root






