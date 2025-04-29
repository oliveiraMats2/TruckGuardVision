#!/usr/bin/env python3
import argparse
from copy import deepcopy
import torch
import yaml

from src.core.yolo.models.yolo import Model
from src.core.yolo.utils.torch_utils import select_device, is_parallel

NUM_CLASSES = 3

def reparameterize(args):
    device = select_device(args.device, batch_size=1)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Create model from deploy configuration
    model = Model(args.deploy_yaml, ch=3, nc=NUM_CLASSES).to(device)

    # Read deploy yaml
    with open(args.deploy_yaml) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    # Prepare intersect state dict
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # reparameterized YOLOR layers
    for i in range((model.nc+5)*anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, :].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, :].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, :].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    # Prepare ckpt for saving
    new_ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
                'optimizer': None,
                'training_results': None,
                'epoch': -1}
                
    # Save reparameterized model
    
    if args.output_path is None:
        args.output_path = args.checkpoint_path.replace('.pt', '_reparam.pt')
    
    torch.save(new_ckpt, args.output_path)
    print(f"Reparameterized model saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Reparameterize YOLOR model")
    parser.add_argument('checkpoint_path', type=str,
                        help="Path to original trained checkpoint")
    parser.add_argument('--output_path', type=str, default=None,
                        help="Output path for saving the reparameterized model")
    parser.add_argument('--deploy_yaml', type=str, default='Src/app/yolo_deploy/cfg/deploy/yolov7.yaml',
                        help="Path to deploy yaml config for reparameterization")
    parser.add_argument('--device', type=str, default='0',
                        help="Device identifier (e.g., 'cpu' or '0' for GPU 0)")
    args = parser.parse_args()

    reparameterize(args)

if __name__ == '__main__':
    main()
