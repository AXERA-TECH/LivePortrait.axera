import argparse
import cv2
import numpy as np
from axengine import InferenceSession
import os
import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os.path as osp
from loguru import logger
from numpy import ndarray
import pickle as pkl
import torch
import torch.nn.functional as F
from cropper import Cropper


appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module = None, None, None, None, None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="LivePortrait",
        description="LivePortrait: A Real-time 3D Live Portrait Animation System"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source image.",
    )
    parser.add_argument(
        "--driving",
        type=str,
        required=True,
        help="Path to driving image.",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Path to onnx models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path to infer results.",
    )
    
    return parser.parse_args()


def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6) -> np.ndarray:
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))


def calc_eye_close_ratio(lmk: np.ndarray, target_eye_ratio: np.ndarray = None) -> np.ndarray:
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)


def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)


def concat_frames(driving_image_lst, source_image_lst, I_p_lst):
    # TODO: add more concat style, e.g., left-down corner driving
    out_lst = []
    h, w, _ = I_p_lst[0].shape
    source_image_resized_lst = [cv2.resize(img, (w, h)) for img in source_image_lst]

    for idx, _ in enumerate(I_p_lst):
        I_p = I_p_lst[idx]
        source_image_resized = source_image_resized_lst[idx] if len(source_image_lst) > 1 else source_image_resized_lst[0]

        if driving_image_lst is None:
            out = np.hstack((source_image_resized, I_p))
        else:
            driving_image = driving_image_lst[idx]
            driving_image_resized = cv2.resize(driving_image, (w, h))
            out = np.hstack((driving_image_resized, source_image_resized, I_p))

        out_lst.append(out)
    return out_lst


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    """ conduct similarity or affine transformation to the image, do not do border operation!
    img:
    M: 2x3 matrix or 3x3 matrix
    dsize: target shape (width, height)
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode, borderValue=(0, 0, 0))
    else:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def prepare_paste_back(mask_crop, crop_M_c2o, dsize):
    """prepare mask for later image paste back
    """
    mask_ori = _transform_img(mask_crop, crop_M_c2o, dsize)
    mask_ori = mask_ori.astype(np.float32) / 255.
    return mask_ori


def paste_back(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image
    """
    dsize = (img_ori.shape[1], img_ori.shape[0])
    result = _transform_img(img_crop, M_c2o, dsize=dsize)
    result = np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
    return result


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            logger.info(f"Make dir: {d}")
    return d


def dct2device(dct: dict, device):
    for key in dct:
        if isinstance(dct[key], torch.Tensor):
            dct[key] = dct[key].to(device)
        else:
            dct[key] = torch.tensor(dct[key]).to(device)
    return dct


PI = np.pi

def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 97.5

        return degree

    return pred


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def load_image_rgb(image_path: str):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_limit(img: np.ndarray, max_dim=1920, division=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = max(division, 1)
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def preprocess(input_data):
    img_rgb = load_image_rgb(input_data)
    img_rgb = resize_to_limit(img_rgb)
    return [img_rgb]


def postprocess(output_data):
    # Implement your postprocessing steps here
    # For example, you might convert the output to a specific format
    return output_data


def infer(model, input_data):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    input_data = preprocess(input_data) # rgb, resize & limit
    result = model.run([output_name], {input_name: input_data})
    return postprocess(result)


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def calc_ratio(lmk_lst):
    input_eye_ratio_lst = []
    input_lip_ratio_lst = []
    for lmk in lmk_lst:
        # for eyes retargeting
        input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
        # for lip retargeting
        input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
    return input_eye_ratio_lst, input_lip_ratio_lst


def prepare_videos(imgs) -> torch.Tensor:
    """ construct the input as standard
    imgs: NxBxHxWx3, uint8
    """
    device = "cpu"
    if isinstance(imgs, list):
        _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
    elif isinstance(imgs, np.ndarray):
        _imgs = imgs
    else:
        raise ValueError(f'imgs type error: {type(imgs)}')

    y = _imgs.astype(np.float32) / 255.
    y = np.clip(y, 0, 1)  # clip to 0~1
    y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
    y = y.to(device)

    return y


def get_kp_info(x: torch.Tensor) -> dict:
    """ get the implicit keypoint information
    x: Bx3xHxW, normalized to 0~1
    flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
    return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
    """
    outs = motion_extractor.run(input_feed={"input": x.numpy()}) # TODO: axengine 中的 run 输入参数与 ort 还是些许不同
    outs = list(outs.values())
    kp_info = {}
    kp_info['pitch'] = torch.from_numpy(outs[0])
    kp_info['yaw'] = torch.from_numpy(outs[1])
    kp_info['roll'] = torch.from_numpy(outs[2])
    kp_info['t'] = torch.from_numpy(outs[3])
    kp_info['exp'] = torch.from_numpy(outs[4])
    kp_info['scale'] = torch.from_numpy(outs[5])
    kp_info['kp'] = torch.from_numpy(outs[6])

    flag_refine_info: bool = True
    if flag_refine_info:
        bs = kp_info['kp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

    return kp_info


def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info['kp']    # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']
    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3), 欧拉角转换为旋转矩阵

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


def make_motion_template(I_lst, c_eyes_lst, c_lip_lst):
    n_frames = I_lst.shape[0]
    template_dct = {
        'n_frames': n_frames,
        'output_fps': 25,
        'motion': [],
        'c_eyes_lst': [],
        'c_lip_lst': [],
    }

    for i in range(n_frames):
        # collect s, R, δ and t for inference
        I_i = I_lst[i]
        x_i_info = get_kp_info(I_i)
        x_s = transform_keypoint(x_i_info)
        R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

        item_dct = {
            'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
            'R': R_i.cpu().numpy().astype(np.float32),
            'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
            't': x_i_info['t'].cpu().numpy().astype(np.float32),
            'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
            'x_s': x_s.cpu().numpy().astype(np.float32),
        }

        template_dct['motion'].append(item_dct)

        c_eyes = c_eyes_lst[i].astype(np.float32)
        template_dct['c_eyes_lst'].append(c_eyes)

        c_lip = c_lip_lst[i].astype(np.float32)
        template_dct['c_lip_lst'].append(c_lip)

    return template_dct


def prepare_source(img: np.ndarray) -> torch.Tensor:
    """ construct the input as standard
    img: HxWx3, uint8, 256x256
    """
    device = "cpu"
    h, w = img.shape[:2]
    x = img.copy()

    if x.ndim == 3:
        x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
    elif x.ndim == 4:
        x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
    else:
        raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
    x = np.clip(x, 0, 1)  # clip to 0~1
    x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
    x = x.to(device)
    return x


def extract_feature_3d(x: torch.Tensor) -> torch.Tensor:
    """ get the appearance feature of the image by F
    x: Bx3xHxW, normalized to 0~1
    """
    outs = appearance_feature_extractor.run(input_feed={"input": x.numpy()})
    outs = list(outs.values())[0]
    return torch.from_numpy(outs)


def stitch(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: BxNx3
    kp_driving: BxNx3
    Return: Bx(3*num_kp+2)
    """
    feat_stiching = concat_feat(kp_source, kp_driving)
    delta = stitching_retargeting_module.run(input_feed={"input": feat_stiching.numpy()})
    delta = list(delta.values())[0]
    return torch.from_numpy(delta)


def stitching(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """ conduct the stitching
    kp_source: Bxnum_kpx3
    kp_driving: Bxnum_kpx3
    """

    bs, num_kp = kp_source.shape[:2]

    kp_driving_new = kp_driving.clone()
    delta = stitch(kp_source, kp_driving_new)

    delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
    delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

    kp_driving_new += delta_exp
    kp_driving_new[..., :2] += delta_tx_ty

    return kp_driving_new


def warp_decode(feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """ get the image after the warping of the implicit keypoints
    feature_3d: Bx32x16x64x64, feature volume
    kp_source: BxNx3
    kp_driving: BxNx3
    """
    outs = warping_module.run([], {"feature_3d": feature_3d.numpy(), "kp_driving": kp_driving.numpy(), "kp_source": kp_source.numpy()})[2]
    # outs = warping_module.run(input_feed={"feature_3d": feature_3d.numpy(), "kp_driving": kp_driving.numpy(), "kp_source": kp_source.numpy()})['out']
    outs = spade_generator.run(input_feed={"input":  outs})
    outs = list(outs.values())[0]
    ret_dct = {}
    ret_dct['out'] = torch.from_numpy(outs)
    return ret_dct


def parse_output(out: torch.Tensor) -> np.ndarray:
    """ construct the output as standard
    return: 1xHxWx3, uint8
    """
    out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
    out = np.clip(out, 0, 1)  # clip to 0~1
    out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

    return out


def load_model(model_type, model_path=None):
    if model_type == 'appearance_feature_extractor':
        model = InferenceSession.load_from_model(f"{model_path}/feature_extractor.axmodel")
    elif model_type == 'motion_extractor':
        model = InferenceSession.load_from_model(f'{model_path}/motion_extractor.axmodel')
    elif model_type == 'warping_module':
        model = ort.InferenceSession(f'{model_path}/warp.onnx', providers=["CPUExecutionProvider"])
        # model = InferenceSession.load_from_model(f'{model_path}/warp.axmodel')
    elif model_type == 'spade_generator':
        model = InferenceSession.load_from_model(f'{model_path}/spade_generator.axmodel')
    elif model_type == 'stitching_retargeting_module':
        model = InferenceSession.load_from_model(f'{model_path}/stitching_retargeting.axmodel')
    return model


def main():
    args = parse_args()

    global appearance_feature_extractor
    appearance_feature_extractor = load_model("appearance_feature_extractor", args.models)

    global motion_extractor
    motion_extractor = load_model("motion_extractor", args.models)

    global warping_module
    warping_module = load_model("warping_module", args.models)

    global spade_generator
    spade_generator = load_model("spade_generator", args.models)

    global stitching_retargeting_module
    stitching_retargeting_module = load_model("stitching_retargeting_module", args.models)

    source = args.source
    driving = args.driving

    source_rgb_lst = preprocess(source)  # rgb, resize & limit
    driving_rgb_lst = [load_image_rgb(driving)] # rgb

    ######## make motion template ########
    cropper: Cropper = Cropper()
    logger.info("Start making driving motion template...")
    driving_n_frames = len(driving_rgb_lst)
    n_frames = driving_n_frames
    driving_lmk_crop_lst = cropper.calc_lmks_from_cropped_video(driving_rgb_lst) # cropper.
    driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
    #######################################

    c_d_eyes_lst, c_d_lip_lst = calc_ratio(driving_lmk_crop_lst)
    # save the motion template
    I_d_lst = prepare_videos(driving_rgb_crop_256x256_lst)
    driving_template_dct = make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst)
    # wfp_template = remove_suffix(args.driving) + '.pkl'
    # dump(wfp_template, driving_template_dct)
    # logger.info(f"Dump motion template to {wfp_template}")

    c_d_eyes_lst = c_d_eyes_lst * n_frames
    c_d_lip_lst = c_d_lip_lst * n_frames

    I_p_pstbk_lst = []
    logger.info("Prepared pasteback mask done.")

    I_p_lst = []
    R_d_0, x_d_0_info = None, None
    flag_normalize_lip = False # inf_cfg.flag_normalize_lip  # not overwrite
    flag_source_video_eye_retargeting = False # inf_cfg.flag_source_video_eye_retargeting  # not overwrite
    lip_delta_before_animation, eye_delta_before_animation = None, None

    ######## process source info ########
    # if the input is a source image, process it only once
    flag_do_crop = True
    if flag_do_crop:
        crop_info = cropper.crop_source_image(source_rgb_lst[0])
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        source_lmk = crop_info['lmk_crop']
        img_crop_256x256 = crop_info['img_crop_256x256']
    else:
        source_lmk = cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
        img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256

    I_s = prepare_source(img_crop_256x256)
    x_s_info = get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = extract_feature_3d(I_s)
    x_s = transform_keypoint(x_s_info)

    # let lip-open scalar to be 0 at first
    mask_crop: ndarray = cv2.imread(make_abs_path('./utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
    mask_ori_float = prepare_paste_back(mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

    with open(make_abs_path('./utils/resources/lip_array.pkl'), 'rb') as f:
        lip_array = pkl.load(f)
    device = "cpu"
    flag_is_source_video = False
    ######## animate ########
    logger.info(f"The output of image-driven portrait animation is an image.")
    for i in range(n_frames):
        x_d_i_info = driving_template_dct['motion'][i]
        x_d_i_info = dct2device(x_d_i_info, device)
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

        if i == 0:  # cache the first frame
            R_d_0 = R_d_i
            x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
        R_new = x_d_r_lst_smooth[i] if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(lip_array).to(dtype=torch.float32, device=device))
        scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
        t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        # Algorithm 1:
        # with stitching and without retargeting
        x_d_i_new = stitching(x_s, x_d_i_new)
        x_d_i_new = x_s + (x_d_i_new - x_s) * 1.0
        out = warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = parse_output(out['out'])[0]
        I_p_lst.append(I_p_i)
        I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
        I_p_pstbk_lst.append(I_p_pstbk)

    mkdir(args.output_dir)
    wfp_concat = None
    ######### build the final concatenation result #########
    # driving frame | source frame | generation
    frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)

    wfp_concat = osp.join(args.output_dir, f'{basename(source)}--{basename(driving)}_concat.jpg')
    cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
    wfp = osp.join(args.output_dir, f'{basename(source)}--{basename(driving)}.jpg')
    cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])

    # final log
    logger.info(f'Animated image: {wfp}')
    logger.info(f'Animated image with concat: {wfp_concat}')


if __name__ == "__main__":
    """
    Usage:
        python3 infer_onnx.py --source ../assets/examples/source/s0.jpg --driving ../assets/examples/driving/d8.jpg --models onnx-models --output-dir output
    """
    main()
