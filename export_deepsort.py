import argparse
import torch
import cv2
from deep_sort_pytorch.deep_sort.deep.model import Net
import numpy as np
import torchvision.transforms as transforms

size = (64, 128)
norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

def _preprocess(im_crops):
    """
    TODO:
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
    """

    def _resize(im, size):
        return cv2.resize(im.astype(np.float32) / 255., size)

    im_batch = torch.cat([norm(_resize(im, size)).unsqueeze(
        0) for im in im_crops], dim=0).float()
    return im_batch

if __name__ == '__main__':
    # available for deepsort .t7
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7')
    opt = parser.parse_args()

    # Input
    img = cv2.imread("dog.jpg", cv2.IMREAD_COLOR)[:, :, (2, 1, 0)]
    img = [img]
    with torch.no_grad():
        img = _preprocess(img)
        img = img.to('cpu')

    # model-------------------------------------------------------------------------------------------------------------
    model = Net(reid=True)
    device = 'cpu'
    state_dict = torch.load(opt.weights, map_location=torch.device(device))['net_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.t7', '.onnx')  # filename
        # model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=['output'])

        # Checks
        print('load onnx model')
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

