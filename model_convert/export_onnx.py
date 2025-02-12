from src.utils.helper import load_model
import torch
import yaml
import argparse
from loguru import logger
import os
import onnx
from onnx import TensorProto, helper
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import numpy as np
import onnxruntime


def onnx_sim(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    logger.info(f"onnx simpilfy successed, and model saved in {onnx_path}")


if __name__ == '__main__':
    
    """
    Usage:
        python3 export_onnx.py -m /path/your/hugging_face/models/LivePortrait/ -o ./onnx-models
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", type=str, help="hugging fance model path")
    parser.add_argument("-o", "--output_dir", type=str, default='./onnx-models', help="onnx model save path")
    args = parser.parse_args()

    model_root = args.model
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_config = yaml.load(open('src/config/models.yaml', 'r'), Loader=yaml.SafeLoader)

    feature_extractor = load_model(f'{model_root}/liveportrait/base_models/appearance_feature_extractor.pth', model_config, 'cpu', 'appearance_feature_extractor', False)
    tmp_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(feature_extractor, tmp_input, f"{output_dir}/feature_extractor.onnx", opset_version=16, verbose=True, 
    input_names=["input"], output_names=['output'])
    onnx_sim(f"{output_dir}/feature_extractor.onnx")

    motion_extractor = load_model(f'{model_root}/liveportrait/base_models/motion_extractor.pth', model_config, 'cpu', 'motion_extractor', False)
    tmp_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(motion_extractor, tmp_input, f"{output_dir}/motion_extractor.onnx", opset_version=16, verbose=True, 
    input_names=["input"]) #output_names=['800', '801', '802', '803', '804', '805', '799']
    onnx_sim(f"{output_dir}/motion_extractor.onnx")

    warping_module = load_model(f'{model_root}/liveportrait/base_models/warping_module.pth', model_config, 'cpu', 'warping_module', False)
    tmp_input = (torch.randn(1, 32, 16, 64, 64), torch.randn(1, 21, 3), torch.randn(1, 21, 3))
    torch.onnx.export(warping_module, tmp_input, f"{output_dir}/warp.onnx", opset_version=16, verbose=True, 
    input_names=["feature_3d", "kp_driving", "kp_source"], output_names=['occlusion_map', 'deformation', 'out'])
    onnx_sim(f"{output_dir}/warp.onnx")

    spade_generator = load_model(f'{model_root}/liveportrait/base_models/spade_generator.pth', model_config, 'cpu', 'spade_generator', False)
    tmp_input = torch.randn(1, 256, 64, 64)
    torch.onnx.export(spade_generator, tmp_input, f"{output_dir}/spade_generator.onnx", opset_version=16, verbose=True, 
    input_names=["input"], output_names=['output'])
    onnx_sim(f"{output_dir}/spade_generator.onnx")

    stitching_retargeting = load_model(f'{model_root}/liveportrait/retargeting_models/stitching_retargeting_module.pth', model_config, 'cpu', 'stitching_retargeting_module', False)
    tmp_input = torch.randn(1, 126)
    torch.onnx.export(stitching_retargeting['stitching'], tmp_input, f"{output_dir}/stitching_retargeting.onnx", opset_version=16, verbose=True, 
    input_names=["input"], output_names=['output'])
    onnx_sim(f"{output_dir}/stitching_retargeting.onnx")
