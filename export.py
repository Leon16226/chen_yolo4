import argparse
import torch
import models
from utils.activations import Mish
from onnxsim import  simplify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/material_7.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[608, 608], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))

    # Load PyTorch model
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        print('load onnx model')
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

