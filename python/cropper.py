from utils.dependencies.insightface.app import FaceAnalysis
from utils.dependencies.insightface.app.common import Face
from utils.timer import Timer
from utils.human_landmark_runner import LandmarkRunner as HumanLandmark
from utils.crop import crop_image
from typing import List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import os.path as osp
import cv2


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj

@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # M_c2o list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces
    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]), reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2]+face['bbox'][0])/2-face_center[0])**2+((face['bbox'][3]+face['bbox'][1])/2-face_center[1])**2)**0.5)
    return faces


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()

    def get(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        face_center = None

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and taskname == 'landmark_2d_106':
                    continue

                # print(f'taskname: {taskname}')
                model.get(img_bgr, face)
            ret.append(face)

        ret = sort_by_direction(ret, direction, face_center)
        return ret

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        print(f'FaceAnalysisDIY warmup time: {elapse:.3f}s')


class Cropper(object):
    def __init__(self, ):
        self.face_analysis_wrapper_provider = ["CPUExecutionProvider"]
        self.insightface_root: str = make_abs_path("./pretrained_weights/insightface")
        self.device_id = 0
        self.landmark_ckpt_path: str = make_abs_path("./pretrained_weights/liveportrait/landmark.onnx")
        self.det_thresh: float = 0.1 # detection threshold
        self.device = "cpu"
        self.image_type = "human_face"
        self.direction: str = "large-small"  # direction of cropping
        self.max_face_num: int = 0  # max face number, 0 mean no limit
        self.dsize: int = 512  # crop size
        self.scale: float = 2.3  # scale factor
        self.vx_ratio: float = 0  # vx ratio
        self.vy_ratio: float = -0.125  # vy ratio +up, -down
        self.flag_do_rot: bool = True # whether to conduct the rotation when flag_do_crop is True

        self.face_analysis_wrapper = FaceAnalysisDIY(
            name="buffalo_l",
            root=self.insightface_root,
            providers=self.face_analysis_wrapper_provider,
        )
        self.face_analysis_wrapper.prepare(ctx_id=self.device_id, det_size=(512, 512), det_thresh=self.det_thresh)
        self.face_analysis_wrapper.warmup()

        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.landmark_ckpt_path,
            onnx_provider=self.device,
            device_id=self.device_id,
        )
        self.human_landmark_runner.warmup()

    def crop_source_image(self, img_rgb_: np.ndarray):
        # crop a source image and get neccessary information
        img_rgb = img_rgb_.copy()  # copy it
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.image_type == "human_face":
            src_face = self.face_analysis_wrapper.get(
                img_bgr,
                flag_do_landmark_2d_106=True,
                direction=self.direction,
                max_face_num=self.max_face_num,
            )

            if len(src_face) == 0:
                log("No face detected in the source image.")
                return None
            elif len(src_face) > 1:
                log(f"More than one face detected in the image, only pick one face by rule {self.direction}.")

            # NOTE: temporarily only pick the first face, to support multiple face in the future
            src_face = src_face[0]
            lmk = src_face.landmark_2d_106  # this is the 106 landmarks from insightface
        else:
            tmp_dct = {
                'animal_face_9': 'animal_face',
                'animal_face_68': 'face'
            }

            img_rgb_pil = Image.fromarray(img_rgb)
            lmk = self.animal_landmark_runner.run(
                img_rgb_pil,
                'face',
                tmp_dct[self.animal_face_type],
                0,
                0
            )

        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=self.dsize,
            scale=self.scale,
            vx_ratio=self.vx_ratio,
            vy_ratio=self.vy_ratio,
            flag_do_rot=self.flag_do_rot,
        )

        # update a 256x256 version for network input
        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite("/data/tmp/yongqiang/LLM/projects/zr/liveportrait_onnx/img_crop.jpg", cv2.cvtColor(ret_dct["img_crop"], cv2.COLOR_BGR2RGB))
        cv2.imwrite("/data/tmp/yongqiang/LLM/projects/zr/liveportrait_onnx/img_crop_256x256.jpg", cv2.cvtColor(ret_dct["img_crop_256x256"], cv2.COLOR_BGR2RGB))
        if self.image_type == "human_face":
            lmk = self.human_landmark_runner.run(img_rgb, lmk)
            ret_dct["lmk_crop"] = lmk
            ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.dsize
        else:
            # 68x2 or 9x2
            ret_dct["lmk_crop"] = lmk

        return ret_dct


    def calc_lmk_from_cropped_image(self, img_rgb_, **kwargs):
        direction = kwargs.get("direction", "large-small")
        src_face = self.face_analysis_wrapper.get(
            contiguous(img_rgb_[..., ::-1]),  # convert to BGR
            flag_do_landmark_2d_106=True,
            direction=direction,
        )
        if len(src_face) == 0:
            log("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            log(f"More than one face detected in the image, only pick one face by rule {direction}.")
        src_face = src_face[0]
        lmk = src_face.landmark_2d_106
        lmk = self.human_landmark_runner.run(img_rgb_, lmk)

        return lmk

    def calc_lmks_from_cropped_video(self, driving_rgb_crop_lst, **kwargs):
        """Tracking based landmarks/alignment"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")

        for idx, frame_rgb_crop in enumerate(driving_rgb_crop_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb_crop[..., ::-1]),  # convert to BGR
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    log(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb_crop, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb_crop, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
        return trajectory.lmk_lst

