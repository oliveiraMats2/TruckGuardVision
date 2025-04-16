# Deploy de modelo YOLOv7 com Triton Server
Conjunto de scripts para realizar o deploy de um modelo YOLOv7 com Triton Server. Para treinar o modelo YOLOv7 e obter os pesos, consulte o [README](Src/core/pragas/yolo/README.md) do módulo `yolo`.

# Pré-requisitos
## Instalação de dependências
Para executar o código deste diretório, é necessário instalar as dependências listadas no arquivo `requirements.txt`. É recomendado o uso de um ambiente virtual como `conda` ou `venv` com *Python 3.9*. Para instalar as dependências, execute o comando abaixo:
```bash
conda create -n pragas_deploy python=3.9 -y
conda activate pragas_deploy
pip install -r Src/app/yolo_deploy/requirements.txt
```

## Preparar modelo para inferência
O script `prepare_model.sh` recebe como argumento o caminho para o modelo treinado e realiza as seguintes operações:
1. Reparametriza o modelo, diminuindo o número de parâmetros e aumentando a velocidade de inferência
2. Converte o modelo para o formato ONNX
3. Converte o modelo para do formato ONNX para o formato TensorRT
4. Move o modelo em formato TensorRT para a pasta `Src/app/yolo_deploy/triton-deploy/models/yolov7/1`, onde o Triton Server irá buscar o modelo.

**A execução deste script é demorada devido à conversão do modelo para o formato TensorRT. Com 4GB de VRAM alocados, o tempo de execução foi de aproximadamente 20 minutos.**

Para executar o script, utilize o comando abaixo:
```bash
./Src/app/yolo_deploy/prepare_model.sh caminho/para/pesos_de_treino.pt
```

# Subir o serviço de inferência
O comando abaixo irá subir o serviço de inferência, que é composto por:
1. Triton Server, que é responsável por servir o modelo de inferência
2. API, que é responsável por receber as imagens e enviar para o Triton Server realizar a inferência. A API é implementada em FastAPI e é disponibilizada na em `http://localhost:8080`.
```bash
sudo docker compose -f Src/app/yolo_deploy/docker-compose.yml up --build
```

# Testar o serviço de inferência
```bash
python Src/app/yolo_deploy/test_inference.py caminho/para/imagem.jpg
```