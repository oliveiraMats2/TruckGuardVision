# YOLO Detection Module
Este módulo contém o código para treinar, validar e executar a detecção de pragas em imagens de armadilhas.

# Pré-requisitos
## Instalação de dependências
Para executar o código deste diretório, é necessário instalar as dependências listadas no arquivo `requirements.txt`. É recomendado o uso de um ambiente virtual como `conda` ou `venv` com *Python 3.9*. Para instalar as dependências, execute o comando abaixo:

```bash
conda create -n pragas python=3.9 -y
conda activate pragas
pip install -r Src/core/pragas/yolo/requirements.txt
```

## Download de pesos pré-treinados
Para treinar o modelo YOLO, é necessário baixar os pesos pré-treinados do modelo YOLOv5. [Acesse o link](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1) e baixe o arquivo `yolov7_training.pt`. Após baixar o arquivo, mova-o para o diretório `weights` na raiz do projeto.


## labelme_converter
Script resonsável por converter as anotações e imagens do formato LabelMe para o formato YOLO. Além disso, também calcula a bounding box de cada anotação a partir dos pontos de polígono fornecidos no LabelMe. Para converter para o formato YOLO, execute o comando abaixo:
```bash
python -m Src.core.pragas.yolo.labelme_converter pragas_onedrive --output-images datasets/pragas/images --output-annotations datasets/pragas/labels --image-dir pragas_onedrive --split 0.7 0.2 0.1 --format yolo
```
O comando acima irá converter as imagens e anotações do diretório `pragas_onedrive` para o formato YOLO, salvando as imagens no diretório `datasets/pragas/images` e as anotações no diretório `datasets/pragas/labels`. Além disso, ele irá dividir as imagens em conjuntos de treino, validação e teste, com 70%, 20% e 10% das imagens, respectivamente.

# Treino
Para treinar o modelo YOLO, execute o comando abaixo:

```bash
python -m Src.core.pragas.yolo.train --workers 8 --device 0 --batch-size 8 --data Src/core/pragas/yolo/data/pragas.yaml --img 640 640 --cfg  Src/core/pragas/yolo/cfg/training/yolov7.yaml --weights weights/yolov7_training.pt --name yolov7-pragas --hyp Src/core/pragas/yolo/data/hyp.scratch.p5.yaml --save_period 1
```

# Teste
Para testar o modelo YOLO, execute o comando abaixo:

```bash
python -m Src.core.pragas.yolo.test --data yolo/data/pragas.yaml --img 640 --batch 2 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7-pragas/weights/best.pt --name yolov7_pragas_640_val
```

# Deploy
Os scripts de deploy estão disponíveis na pasta [yolo_deploy](Src/app/yolo_deploy/README.md).

# python -m src.yolo.train --workers 8 --device 0 --batch-size 1 --data yolo/data/pragas.yaml --img 640 640 --cfg  yolo/cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7-pragas --hyp pragas/yolo/data/coco.yaml --save_period 1
