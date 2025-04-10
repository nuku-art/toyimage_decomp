FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo 
RUN apt -y update && apt -y upgrade \
    libopencv-dev libgl1-mesa-dev wget && \
    wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.3.340/quarto-1.3.340-linux-amd64.deb && \
    dpkg -i quarto-1.3.340-linux-amd64.deb && \
    apt-get install -f && \
    pip install jupyterlab ipykernel opencv-python pandas numpy scikit-learn albumentations matplotlib h5py seaborn captum 




