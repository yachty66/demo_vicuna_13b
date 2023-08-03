# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime


WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install wget
RUN apt-get update && apt-get install -y wget

# Download the wheel file
RUN wget https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.2/auto_gptq-0.2.2+cu117-cp310-cp310-linux_x86_64.whl

# Install the wheel file
RUN pip3 install auto_gptq-0.2.2+cu117-cp310-cp310-linux_x86_64.whl

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py