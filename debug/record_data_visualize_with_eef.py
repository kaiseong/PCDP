"""
모델 추론으로 저장된 obs zarr데이터들
'한 장면씩 넘겨보는' 디버깅 + (그리퍼 좌표계 + 좌/우 턱 ROI) 시각화
- 로봇 포즈: 로봇 좌표계 (Robot base)
- 포인트클라우드: 카메라→플랫폼(Base) 변환 후, 시각화 시 Base→Robot으로 변환하여 동일 좌표계에서 표시
- ROI는 그리퍼 프레임(TCP)에만 부착 + gripper_width에 따라 매 프레임 갱신 (턱 '안쪽 면' 기준)
- N: 다음, B: 이전, Q: 종료
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import open3d as o3d
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.common import RISE_transformation as rise_tf


robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.295],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
], dtype=np.float64)

# 카메라→플랫폼(Base)
camera_to_base = np.array([
    [  0.007131,  -0.91491 ,   0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656 , -0.00918],
    [ -0.025717,  -0.403641,  -0.914552,  0.50821],
    [  0.      ,   0.      ,   0.      ,  1.     ]
], dtype=np.float64)

workspace_bounds = np.array([
    [ 0.000, 0.715],   # X (m)
    [-0.400, 0.350],   # Y (m)
    [-0.100, 0.400],   # Z (m)
], dtype=np.float64)

# Base→Robot (시각화용)
base_to_robot = np.linalg.inv(robot_to_base)

# ==================== GRIPPER / ROI 파라미터 ====================
# 플랜지 EEF → 그리퍼 EEF(TCP) 고정 변환(장착 오프셋) — 하드웨어에 맞게 튜닝
EEF_TO_GRIP_TRANSLATION = np.array([0.0, 0.0, 0.12], dtype=np.float64)  # [mx, my, mz]
EEF_TO_GRIP_ROT_Z_DEG   = 0.0  # EEF→GRIP 장착 회전(필요시 조정)

# 그리퍼 프레임에서의 추가 미세 오프셋/회전(시각화 정렬용)
GRIP_LOCAL_OFFSET        = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ROTATE_AROUND_GRIP_Z_DEG = 90.0  # 필요하면 ±90 등

# ROI(좌/우 턱) 정의 (그리퍼 프레임 로컬 축 기준)
ROI_CLOSE_AXIS    = 'x'   # 닫힘축: 'x'|'y'|'z'
ROI_APPROACH_AXIS = 'y'   # 접근축(닫힘축과 달라야 함)
ROI_THICKNESS = 0.050     # 각 턱 박스 두께 (닫힘축 방향, m)
ROI_DEPTH     = 0.030     # 접근축 길이 (m)
ROI_HEIGHT    = 0.050     # 나머지 축 길이 (m)
ROI_CENTER_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# gripper_width → ROI_GAP 매핑(필요 시 스케일/오프셋 보정)
USE_WIDTH_FOR_GAP     = True
WIDTH_TO_GAP_SCALE    = 0.001    # gap = scale * width + offset
WIDTH_TO_GAP_OFFSET   = 0.0
ROI_MIN_GAP           = 0.010  # 안전한 최소 간격
ROI_MAX_GAP           = 0.150  # 물리적 최대 간격
SMOOTH_GAP_ALPHA      = 0.0    # 0.0이면 스무딩 안함, 0.3~0.7 추천

SHOW_ROI       = True
SHOW_GRIP_AXIS = True
AXIS_SIZE_ROBOT= 0.10
AXIS_SIZE_GRIP = 0.06

# ==================== State Management ====================
class VisController:
    """Manages visualization state and key inputs."""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.vis_changed = True

    def step_forward(self, vis):
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self.vis_changed = True
            print(f"Frame: {self.current_step}/{self.total_steps - 1}")
        return False

    def step_backward(self, vis):
        if self.current_step > 0:
            self.current_step -= 1
            self.vis_changed = True
            print(f"Frame: {self.current_step}/{self.total_steps - 1}")
        return False


# ==================== Helpers ====================
def transform_points(T_4x4, xyz):
    """xyz(Base) -> xyz'(Target) with homogeneous transform T."""
    if xyz.size == 0:
        return xyz
    N = xyz.shape[0]
    xyz_h = np.c_[xyz, np.ones((N, 1), dtype=np.float64)]  # (N,4)
    xyz_t = (T_4x4 @ xyz_h.T).T[:, :3]
    return xyz_t

def axis_index(ax: str) -> int:
    return {'x':0,'y':1,'z':2}[ax.lower()]

def rotz(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=np.float64)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def make_box_lineset(center_local, extent_xyz, color):
    """(그리퍼 로컬) 박스 와이어프레임 LineSet 생성"""
    ex, ey, ez = extent_xyz
    hx, hy, hz = ex/2, ey/2, ez/2
    corners = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[-hx,+hy,-hz],[+hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[-hx,+hy,+hz],[+hx,+hy,+hz],
    ], dtype=np.float64)
    corners += center_local.reshape(1,3)
    lines = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color,float),(len(lines),1)))
    return ls

def box_corners_local(center_local, extent_xyz):
    """로컬 기준 박스 8코너 반환 (N=8,3)"""
    ex, ey, ez = extent_xyz
    hx, hy, hz = ex/2, ey/2, ez/2
    corners = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[-hx,+hy,-hz],[+hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[-hx,+hy,+hz],[+hx,+hy,+hz],
    ], dtype=np.float64)
    return corners + center_local.reshape(1,3)

def set_lineset_points_world(ls: o3d.geometry.LineSet, corners_local, T_world_from_local):
    """라인셋의 points를 '월드 좌표'로 직접 갱신"""
    N = corners_local.shape[0]
    corners_h = np.c_[corners_local, np.ones((N,1), dtype=np.float64)]
    corners_world = (T_world_from_local @ corners_h.T).T[:, :3]
    ls.points = o3d.utility.Vector3dVector(corners_world)

def points_in_roi_local(points_local, center_local, extent_xyz):
    ex, ey, ez = extent_xyz; hx, hy, hz = ex/2, ey/2, ez/2
    d = points_local - center_local.reshape(1,3)
    return (np.abs(d[:,0]) <= hx) & (np.abs(d[:,1]) <= hy) & (np.abs(d[:,2]) <= hz)

def width_to_gap(width, last_gap=None):
    """gripper_width → ROI_GAP 변환(+클램프, 옵션 스무딩). width가 NaN이면 last_gap이나 기본값 사용."""
    if not USE_WIDTH_FOR_GAP:
        gap = 0.090
    else:
        if width is None or np.isnan(width):
            gap = 0.090 if last_gap is None else last_gap
        else:
            gap = WIDTH_TO_GAP_SCALE * float(width) + WIDTH_TO_GAP_OFFSET
    gap = float(np.clip(gap, ROI_MIN_GAP, ROI_MAX_GAP))
    if last_gap is not None and SMOOTH_GAP_ALPHA > 0.0:
        gap = SMOOTH_GAP_ALPHA * last_gap + (1.0 - SMOOTH_GAP_ALPHA) * gap
    return gap

def get_roi_centers_from_gap(gap, thickness, close_axis_idx, center_offset, anchor="inner_face"):
    """
    anchor:
      - "inner_face": 각 턱의 '안쪽 면'에서 중심 방향으로 thickness/2 만큼 이동한 지점이 박스 중심
                      (즉, 박스가 '턱 안쪽 면→중심'을 채움)
      - "centered":   양 턱의 중심을 gap/2 기준으로 대칭 배치
    """
    offs_centered = gap * 0.5 - thickness * 0.5
    cL = center_offset.copy()
    cR = center_offset.copy()
    if anchor == "inner_face":
        cL[close_axis_idx] = -gap * 0.5 + thickness * 0.5
        cR[close_axis_idx] = +gap * 0.5 - thickness * 0.5
    else:  # "centered"
        cL[close_axis_idx] = -offs_centered
        cR[close_axis_idx] = +offs_centered
    return cL, cR


def interactive_visualize_with_gripper(obs_episode):
    """
    Visualizes point cloud & GRIPPER pose in Robot base with N/B navigation.
    - pointcloud: Camera->Base->Robot
    - T_robot_grip_vis: Robot<-Grip (TCP)
    - ROI: Grip-local boxes; every frame we rebuild world points (moves with gripper + width)
    """
    pts_seq  = obs_episode['pointcloud']
    pose_seq = obs_episode['robot_eef_pose']  # [T,6], m/rad (Robot base 기준 플랜지)
    if 'robot_gripper' in obs_episode:
        grip_width  = obs_episode['robot_gripper'][:, 0]  # width (m)
        grip_effort = obs_episode['robot_gripper'][:, 1]
    else:
        grip_width = None
        grip_effort = None

    num_steps = len(pts_seq)
    if num_steps == 0:
        print("No data to visualize."); return

    preprocess = PointCloudPreprocessor(
        extrinsics_matrix=camera_to_base,
        workspace_bounds=workspace_bounds,
        enable_sampling=False,
        enable_transform=True,
        enable_filter=True,
        nb_points=10, sor_std=1.7,
    )

    # --- O3D window & callbacks
    controller = VisController(num_steps)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Episode Inspector (Robot + Grip ROI) [N/B/Q]", width=1280, height=720)
    is_running = True
    def cb_exit(v): 
        nonlocal is_running; is_running = False; return False
    vis.register_key_callback(ord("N"), controller.step_forward)
    vis.register_key_callback(ord("B"), controller.step_backward)
    vis.register_key_callback(ord("Q"), cb_exit)

    # --- Geometries
    pcd = o3d.geometry.PointCloud()
    robot_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_ROBOT)
    vis.add_geometry(robot_origin, reset_bounding_box=False)
    grip_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_GRIP) if SHOW_GRIP_AXIS else None

    # ROI axis index & extents (fixed)
    close_i    = axis_index(ROI_CLOSE_AXIS)
    approach_i = axis_index(ROI_APPROACH_AXIS)
    other_i    = ({0,1,2} - {close_i, approach_i}).pop()
    extent = np.zeros(3, dtype=np.float64)
    extent[close_i]    = ROI_THICKNESS
    extent[approach_i] = ROI_DEPTH
    extent[other_i]    = ROI_HEIGHT

    # Empty LineSets; we’ll fill points each frame
    roi_left  = make_box_lineset(np.zeros(3), extent, color=[1.0, 0.2, 0.2])  # 빨강=왼쪽(닫힘축 -)
    roi_right = make_box_lineset(np.zeros(3), extent, color=[0.2, 0.4, 1.0])  # 파랑=오른쪽(닫힘축 +)

    # Fixed transforms
    T_eef_to_grip       = make_T(rotz(np.deg2rad(EEF_TO_GRIP_ROT_Z_DEG)), EEF_TO_GRIP_TRANSLATION)
    T_grip_local_offset = make_T(np.eye(3), GRIP_LOCAL_OFFSET)
    T_grip_local_rotZ   = make_T(rotz(np.deg2rad(ROTATE_AROUND_GRIP_Z_DEG)), np.zeros(3))

    print("\n=== Keys ===\n[N] Next  [B] Back  [Q] Quit\n")

    # State
    first = True
    T_prev_grip = np.eye(4, dtype=np.float64)
    last_gap = None
    ANCHOR_MODE = "inner_face"   # <-- 요구사항 반영: 턱 '안쪽 면' 기준

    while is_running:
        if controller.vis_changed:
            step = controller.current_step

            # --- Point cloud: Cam->Base (preprocess), then Base->Robot
            pc_base = preprocess(pts_seq[step])
            if pc_base.size == 0:
                controller.vis_changed = False
                vis.poll_events(); vis.update_renderer(); time.sleep(0.01); continue
            xyz_base = pc_base[:, :3].astype(np.float64)
            rgb = pc_base[:, 3:6].astype(np.float64)
            if rgb.max() > 1.0: rgb = rgb/255.0
            xyz_robot = transform_points(base_to_robot, xyz_base)
            pcd.points = o3d.utility.Vector3dVector(xyz_robot)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            # --- Robot<-EEF (from dataset), then ->Grip (TCP)
            trans = pose_seq[step][:3]
            rpy   = pose_seq[step][3:6]
            T_robot_eef  = rise_tf.rot_trans_mat(trans, rpy)      # Robot<-EEF
            T_robot_grip = T_robot_eef @ T_eef_to_grip            # Robot<-Grip
            T_robot_grip_vis = T_robot_grip @ T_grip_local_offset @ T_grip_local_rotZ

            # --- gap from width (그리퍼 닫힘에 따라 gap 변화)
            width_val = float(grip_width[step]) if grip_width is not None else np.nan
            effort_val = float(grip_effort[step]) if grip_effort is not None else np.nan
            gap_t = width_to_gap(width_val, last_gap)
            last_gap = gap_t

            # --- 프레임별 ROI 로컬 중심(양 턱 '안쪽 면' 기준)
            c_left_t, c_right_t = get_roi_centers_from_gap(
                gap=gap_t,
                thickness=ROI_THICKNESS,
                close_axis_idx=close_i,
                center_offset=ROI_CENTER_OFFSET,
                anchor=ANCHOR_MODE
            )

            # --- ROI 라인셋: 이번 프레임의 '월드 좌표' 포인트로 갱신
            corners_L_local = box_corners_local(c_left_t,  extent)
            corners_R_local = box_corners_local(c_right_t, extent)
            set_lineset_points_world(roi_left,  corners_L_local, T_robot_grip_vis)
            set_lineset_points_world(roi_right, corners_R_local, T_robot_grip_vis)

            if first:
                vis.add_geometry(pcd, reset_bounding_box=True)
                if SHOW_GRIP_AXIS and grip_axis is not None:
                    grip_axis.transform(T_robot_grip_vis)
                    vis.add_geometry(grip_axis, reset_bounding_box=False)
                if SHOW_ROI:
                    vis.add_geometry(roi_left,  reset_bounding_box=False)
                    vis.add_geometry(roi_right, reset_bounding_box=False)

                # View
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                ctr.set_front([0.0, 0.0, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.45)

                T_prev_grip = T_robot_grip_vis.copy()
                first = False
            else:
                vis.update_geometry(pcd)

                # 그리퍼 좌표축은 델타로 이동(형상 불변)
                if SHOW_GRIP_AXIS and grip_axis is not None:
                    delta_grip = T_robot_grip_vis @ np.linalg.inv(T_prev_grip)
                    grip_axis.transform(delta_grip)
                    vis.update_geometry(grip_axis)

                # ROI는 월드 points를 이미 새로 넣었으니 update만
                if SHOW_ROI:
                    vis.update_geometry(roi_left)
                    vis.update_geometry(roi_right)

                T_prev_grip = T_robot_grip_vis.copy()

            # --- ROI occupancy in Grip-local for S
            Rg = T_robot_grip_vis[:3, :3]; tg = T_robot_grip_vis[:3, 3]
            pts_local = (xyz_robot - tg) @ Rg.T
            mask_L = points_in_roi_local(pts_local, c_left_t,  extent)
            mask_R = points_in_roi_local(pts_local, c_right_t, extent)
            N_L, N_R = int(mask_L.sum()), int(mask_R.sum())
            S = min(N_L, N_R)
            print(f"[{step}/{num_steps-1}] width={width_val:.4f}m → gap={gap_t:.4f}m | N_L={N_L}, N_R={N_R}, S={S}")
            print(f"effort: {effort_val}")

            controller.vis_changed = False

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()


# ==================== I/O 래퍼 ====================
class EpisodeAnalyzer:
    """에피소드별 저장된 PCDP 데이터 분석 클래스"""
    def __init__(self, recorder_data_dir):
        self.recorder_data_dir = Path(recorder_data_dir)
        self.episodes = self._discover_episodes()

    def _discover_episodes(self):
        episodes = []
        if self.recorder_data_dir.exists():
            for item in sorted(self.recorder_data_dir.iterdir()):
                if item.is_dir() and item.name.startswith('episode_'):
                    episodes.append(item.name)
        return episodes

    def load_episode(self, episode_name):
        episode_dir = self.recorder_data_dir / episode_name
        obs_path = episode_dir / 'obs_replay_buffer.zarr'
        action_path = episode_dir / 'action_replay_buffer.zarr'
        if not obs_path.exists() or not action_path.exists():
            raise FileNotFoundError(f"Episode data not found: {episode_dir}")
        obs_buffer = ReplayBuffer.copy_from_path(str(obs_path), backend='numpy')
        action_buffer = ReplayBuffer.copy_from_path(str(action_path), backend='numpy')
        return obs_buffer, action_buffer


def analyze_episode_quality(obs_buffer, action_buffer, episode_name):
    print(f"\n=== {episode_name} 품질 분석 ===")
    print(f"Observation 스텝: {obs_buffer.n_steps}")
    print(f"Action 스텝: {action_buffer.n_steps}")
    obs_episode = obs_buffer.get_episode(0)
    interactive_visualize_with_gripper(obs_episode)


if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    BASE_DATA_DIR = "/home/nscl/diffusion_policy/data/please_please/recorder_data"
    EPISODE_TO_LOAD = "episode_0163"
    # --------------------------
    try:
        analyzer = EpisodeAnalyzer(BASE_DATA_DIR)
        obs_buffer, action_buffer = analyzer.load_episode(EPISODE_TO_LOAD)
        analyze_episode_quality(obs_buffer, action_buffer, EPISODE_TO_LOAD)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback
        print("An unexpected error occurred:")
        traceback.print_exc()
