# pc_to_depth_test.py
"""
모델 추론으로 저장된 obs zarr데이터들
'한 장면씩 넘겨보는' 디버깅 + (그리퍼 좌표계 + 좌/우 턱 ROI) 시각화
+ [추가] 매 프레임 포인트를 Base→Camera로 되돌려 Depth 이미지로 투영/시각화 (C2D 가정)
+ [추가] 현재 전용 Depth / 누적 Depth / (현재 제외) 누적 Depth 세 창 동시 표시
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import cv2
import open3d as o3d
import torch
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.common import RISE_transformation as rise_tf

# ==================== 시각화 옵션 (하드코딩) ====================
SHOW_WORKSPACE = True
SHOW_PLATFORM_FRAME = True
SHOW_CAMERA_FRAME = True
SHOW_EEF_FRAME = True

# ==================== 카메라/좌표계 파라미터 ====================
# 로봇↔베이스
robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.295],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
], dtype=np.float64)

# 카메라→플랫폼(Base) (Orbbec Depth 카메라라고 가정)
camera_to_base = np.array([
    [  0.007131,  -0.91491 ,   0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656 , -0.00918],
    [ -0.025717,  -0.403641,  -0.914552,  0.50821],
    [  0.      ,   0.      ,   0.      ,  1.     ]
], dtype=np.float64)

workspace_bounds = np.array([
    [ 0.132, 0.715],   # X (m)
    [-0.400, 0.350],   # Y (m)
    [-0.100, 0.500],   # Z (m)
], dtype=np.float64)

# 변환 행렬들
base_to_robot  = np.linalg.inv(robot_to_base)
base_to_camera = np.linalg.inv(camera_to_base)  # ← 역변환: Base→Camera

# Depth 카메라 intrinsics (C2D 기준)
DEPTH_W, DEPTH_H = 320, 288
fx_d = 252.69204711914062
fy_d = 252.65277099609375
cx_d = 166.12030029296875
cy_d = 176.21173095703125
K_depth = np.array([[fx_d, 0., cx_d],
                    [0.,  fy_d, cy_d],
                    [0.,  0.,   1. ]], dtype=np.float64)
DIST_DEPTH = None  # rectified 가정. raw라면 OpenCV rational 8계수로 교체

# ==================== GRIPPER / ROI 파라미터 ====================
EEF_TO_GRIP_TRANSLATION = np.array([0.0, 0.0, 0.12], dtype=np.float64)
EEF_TO_GRIP_ROT_Z_DEG   = 0.0
GRIP_LOCAL_OFFSET        = np.array([0.0, 0.0, 0.0], dtype=np.float64)
ROTATE_AROUND_GRIP_Z_DEG = 90.0

ROI_CLOSE_AXIS    = 'x'
ROI_APPROACH_AXIS = 'y'
ROI_THICKNESS = 0.000
ROI_DEPTH     = 0.000
ROI_HEIGHT    = 0.000
ROI_CENTER_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

USE_WIDTH_FOR_GAP     = True
WIDTH_TO_GAP_SCALE    = 0.001
WIDTH_TO_GAP_OFFSET   = 0.0
ROI_MIN_GAP           = 0.010
ROI_MAX_GAP           = 0.150
SMOOTH_GAP_ALPHA      = 0.0

SHOW_ROI        = False
SHOW_GRIP_AXIS  = False
AXIS_SIZE_ROBOT = 0.001
AXIS_SIZE_GRIP  = 0.06

# ★ 변경: 단일 창 대신 3개 Depth 창을 띄운다 (총 4뷰: Open3D + 3 Depth)
SHOW_DEPTH_WIN   = True
SHOW_DEPTH_CUR   = True   # 현재 프레임만
SHOW_DEPTH_FUSED = True   # 누적(fused, 현재 포함)
SHOW_DEPTH_MEMO  = True   # 누적-현재 (현재 제외 메모리만)

# ==================== 시각화 컨트롤 ====================
class VisController:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.vis_changed = True

    def step_forward(self, vis):
        if self.current_step < self.total_steps - 1:
            self.current_step += 3
            self.vis_changed = True
            print(f"Frame: {self.current_step}/{self.total_steps - 1}")
        return False

    def step_backward(self, vis):
        if self.current_step > 0:
            self.current_step -= 3
            self.vis_changed = True
            print(f"Frame: {self.current_step}/{self.total_steps - 1}")
        return False

# ==================== Helper 함수들 ====================
def transform_points(T_4x4, xyz):
    if xyz.size == 0:
        return xyz
    N = xyz.shape[0]
    xyz_h = np.c_[xyz, np.ones((N, 1), dtype=np.float64)]
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
    ex, ey, ez = extent_xyz
    hx, hy, hz = ex/2, ey/2, ez/2
    corners = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[-hx,+hy,-hz],[+hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[-hx,+hy,+hz],[+hx,+hy,+hz],
    ], dtype=np.float64)
    return corners + center_local.reshape(1,3)

def set_lineset_points_world(ls: o3d.geometry.LineSet, corners_local, T_world_from_local):
    N = corners_local.shape[0]
    corners_h = np.c_[corners_local, np.ones((N,1), dtype=np.float64)]
    corners_world = (T_world_from_local @ corners_h.T).T[:, :3]
    ls.points = o3d.utility.Vector3dVector(corners_world)

def points_in_roi_local(points_local, center_local, extent_xyz):
    ex, ey, ez = extent_xyz; hx, hy, hz = ex/2, ey/2, ez/2
    d = points_local - center_local.reshape(1,3)
    return (np.abs(d[:,0]) <= hx) & (np.abs(d[:,1]) <= hy) & (np.abs(d[:,2]) <= hz)

def width_to_gap(width, last_gap=None):
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
    offs_centered = gap * 0.5 - thickness * 0.5
    cL = center_offset.copy()
    cR = center_offset.copy()
    if anchor == "inner_face":
        cL[close_axis_idx] = -gap * 0.5 + thickness * 0.5
        cR[close_axis_idx] = +gap * 0.5 - thickness * 0.5
    else:
        cL[close_axis_idx] = -offs_centered
        cR[close_axis_idx] = +offs_centered
    return cL, cR

# -------------------- 3D→Depth 래스터라이즈 --------------------
def project_points_to_depth(points_cam_xyz, width, height, K, dist=None, assume_z_in_m=True):
    """
    points_cam_xyz: (N,3) 카메라 좌표계 (C2D 기준 Depth 카메라)
    반환: uint16 depth 이미지(mm), shape=(H,W)
    """
    if points_cam_xyz.size == 0:
        return np.zeros((height, width), dtype=np.uint16)

    # 앞쪽(Z>0)만
    pts = points_cam_xyz[points_cam_xyz[:,2] > 0.0]
    if pts.size == 0:
        return np.zeros((height, width), dtype=np.uint16)

    rvec = np.zeros((3,1), dtype=np.float64)
    tvec = np.zeros((3,1), dtype=np.float64)
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    imgpts, _ = cv2.projectPoints(pts.reshape(-1,1,3), rvec, tvec, K, dist)
    uv = imgpts.reshape(-1,2)
    u = np.rint(uv[:,0]).astype(np.int32)
    v = np.rint(uv[:,1]).astype(np.int32)

    inb = (u>=0) & (u<width) & (v>=0) & (v<height)
    u, v, z = u[inb], v[inb], pts[inb,2]

    # 깊이 단위 변환: meter→mm
    z_mm = (z * 1000.0) if assume_z_in_m else z
    depth = np.zeros((height, width), dtype=np.uint16)

    # Z-buffer (가까운 점 우선: 최소 z)
    lin = (v * width + u).astype(np.int64)
    order = np.argsort(lin)
    lin_s = lin[order]
    z_s   = z_mm[order]
    uniq, first = np.unique(lin_s, return_index=True)
    mins = np.minimum.reduceat(z_s, first)
    mins = np.clip(mins, 0, 65535).astype(np.uint16)
    depth.ravel()[uniq] = mins
    return depth

def depth_to_color(depth_u16, colormap=cv2.COLORMAP_TURBO, p_lo=5, p_hi=95, text=None):
    h, w = depth_u16.shape
    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    scaled = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid):
        p = np.percentile(depth[valid], [p_lo, p_hi]).astype(np.float32)
        lo, hi = float(p[0]), float(p[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(depth[valid].min()), float(depth[valid].max())
        d_clip = np.clip(depth, lo, hi)
        scaled[valid] = ((d_clip[valid] - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(scaled, colormap)
    if text:
        cv2.putText(color, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return color

# ★ voxel key 도우미 (fused와 current를 동일 voxel로 비교)
def voxel_keys_from_xyz(xyz: np.ndarray, voxel_size: float):
    if xyz.size == 0:
        return np.empty((0,), dtype=np.dtype((np.void, 12)))
    grid = np.floor(xyz / float(voxel_size)).astype(np.int32, copy=False)
    grid = np.ascontiguousarray(grid)
    return grid.view(np.dtype((np.void, 12))).ravel()

# ==================== 메인 시각화 루프 ====================
def interactive_visualize_with_gripper(obs_episode):
    """
    Visualizes point cloud & GRIPPER pose in Robot base (Open3D)
    + 같은 프레임의 Depth 이미지 (OpenCV)
    - pointcloud: Camera->Base (preprocess), then Base->Robot (Open3D), Base->Camera (OpenCV Depth)
    - ROI: Grip-local boxes; moves per frame with gripper & width
    - Depth(Window 3개): (1) current-only, (2) fused-all(현재 포함), (3) mem-only(fused-current)
    """
    pts_seq  = obs_episode['pointcloud']          # (T, N, C)
    pose_seq = obs_episode['robot_eef_pose']      # [T,6], m/rad
    if 'robot_gripper' in obs_episode:
        grip_width  = obs_episode['robot_gripper'][:, 0]
        grip_effort = obs_episode['robot_gripper'][:, 1]
    else:
        grip_width = None
        grip_effort = None

    num_steps = len(pts_seq)
    if num_steps == 0:
        print("No data to visualize."); return

    # 누적(fused) 파이프라인: temporal ON (GPU 필요)
    preprocess_fused = PointCloudPreprocessor(
        extrinsics_matrix=camera_to_base,
        workspace_bounds=workspace_bounds,
        enable_sampling=False,
        enable_transform=True,
        enable_filter=True,
        enable_cropping=True,
        nb_points=10,
        sor_std=1.7,
        enable_temporal=False,
        export_mode="fused",
        use_cuda=torch.cuda.is_available(),
        verbose=True,
        temporal_decay=0.95,
        depth_width=DEPTH_W, depth_height=DEPTH_H,
        K_depth=K_depth, dist_depth=DIST_DEPTH
    )
    # 현재 프레임 전용(current): temporal OFF, occl_prune OFF (GPU 의존 제거)
    preprocess_cur = PointCloudPreprocessor(
        extrinsics_matrix=camera_to_base,
        workspace_bounds=workspace_bounds,
        enable_sampling=False,
        enable_transform=True,
        enable_filter=True,
        nb_points=10,
        sor_std=1.7,
        enable_temporal=False,
        export_mode="off",
        enable_occlusion_prune=False,
        use_cuda=False,
        verbose=False,
        depth_width=DEPTH_W, depth_height=DEPTH_H,
        K_depth=K_depth, dist_depth=DIST_DEPTH
    )

    controller = VisController(num_steps)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Episode Inspector (Robot + Grip ROI) [N/B/Q]", width=1280, height=720)
    is_running = True
    def cb_exit(v):
        nonlocal is_running; is_running = False; return False
    vis.register_key_callback(ord("N"), controller.step_forward)
    vis.register_key_callback(ord("B"), controller.step_backward)
    vis.register_key_callback(ord("Q"), cb_exit)

    pcd = o3d.geometry.PointCloud()
    robot_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_ROBOT)
    vis.add_geometry(robot_origin, reset_bounding_box=False)
    grip_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE_GRIP) if SHOW_GRIP_AXIS else None
    eef_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) if SHOW_EEF_FRAME else None


    # ROI 준비
    close_i    = axis_index(ROI_CLOSE_AXIS)
    approach_i = axis_index(ROI_APPROACH_AXIS)
    other_i    = ({0,1,2} - {close_i, approach_i}).pop()
    extent = np.zeros(3, dtype=np.float64)
    extent[close_i]    = ROI_THICKNESS
    extent[approach_i] = ROI_DEPTH
    extent[other_i]    = ROI_HEIGHT

    roi_left  = make_box_lineset(np.zeros(3), extent, color=[1.0, 0.2, 0.2])
    roi_right = make_box_lineset(np.zeros(3), extent, color=[0.2, 0.4, 1.0])

    # 고정 변환
    T_eef_to_grip       = make_T(rotz(np.deg2rad(EEF_TO_GRIP_ROT_Z_DEG)), EEF_TO_GRIP_TRANSLATION)
    T_grip_local_offset = make_T(np.eye(3), GRIP_LOCAL_OFFSET)
    T_grip_local_rotZ   = make_T(rotz(np.deg2rad(ROTATE_AROUND_GRIP_Z_DEG)), np.zeros(3))

    # Workspace and coordinate frames visualization
    if SHOW_WORKSPACE:
        ws_min = workspace_bounds[:, 0]
        ws_max = workspace_bounds[:, 1]
        ws_center = (ws_min + ws_max) / 2
        ws_extent = ws_max - ws_min
        workspace_box = make_box_lineset(ws_center, ws_extent, color=[0.5, 0.5, 1.0])
        workspace_box.transform(base_to_robot)
        vis.add_geometry(workspace_box, reset_bounding_box=False)

    if SHOW_PLATFORM_FRAME:
        platform_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
        platform_frame.transform(base_to_robot)
        vis.add_geometry(platform_frame, reset_bounding_box=False)

    if SHOW_CAMERA_FRAME:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0,0,0])
        T_robot_from_camera = base_to_robot @ camera_to_base
        camera_frame.transform(T_robot_from_camera)
        vis.add_geometry(camera_frame, reset_bounding_box=False)


    print("\n=== Keys ===\n[N] Next  [B] Back  [Q] Quit\n")

    first = True
    T_prev_grip = np.eye(4, dtype=np.float64)
    T_prev_eef_grip = np.eye(4, dtype=np.float64)
    last_gap = None
    ANCHOR_MODE = "inner_face"

    # Depth 창 생성
    if SHOW_DEPTH_WIN:
        if SHOW_DEPTH_CUR:
            cv2.namedWindow("Depth (Now PointCloud)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth (Now PointCloud)", 640, 576)
        if SHOW_DEPTH_FUSED:
            cv2.namedWindow("Depth (Stack PointCloud)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth (Stack PointCloud)", 640, 576)
        if SHOW_DEPTH_MEMO:
            cv2.namedWindow("Depth (Diff PointCloud)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth (Diff PointCloud)", 640, 576)

    while is_running:
        if controller.vis_changed:
            step = controller.current_step

            # --- 현재 프레임 전용: Cam->Base (stateless) ----
            pc_base_cur = preprocess_cur(pts_seq[step])   # (Nc,6) or zero
            # --- 누적(fused): Cam->Base ----
            pc_base_fused = preprocess_fused(pts_seq[step])  # (Nf,7) or zero

            if pc_base_fused.size == 0 and pc_base_cur.size == 0:
                controller.vis_changed = False
                vis.poll_events(); vis.update_renderer(); time.sleep(0.01)
                if SHOW_DEPTH_WIN: cv2.waitKey(1)
                continue

            # Open3D 표시용 (fused 우선)
            xyz_base_fused = pc_base_fused[:, :3].astype(np.float64) if pc_base_fused.size else np.zeros((0,3))
            rgb_fused = pc_base_fused[:, 3:6].astype(np.float64) if pc_base_fused.size else np.zeros((0,3))
            if rgb_fused.size and rgb_fused.max() > 1.0: rgb_fused = rgb_fused/255.0

            xyz_robot = transform_points(base_to_robot, xyz_base_fused)
            pcd.points = o3d.utility.Vector3dVector(xyz_robot)
            pcd.colors = o3d.utility.Vector3dVector(rgb_fused if rgb_fused.size else np.zeros_like(xyz_robot))

            # --- 그리퍼 좌표계/ROI ----
            trans = pose_seq[step][:3]
            rpy   = pose_seq[step][3:6]
            T_robot_eef  = rise_tf.rot_trans_mat(trans, rpy)
            T_robot_grip = T_robot_eef @ T_eef_to_grip
            T_robot_grip_vis = T_robot_grip @ T_grip_local_offset @ T_grip_local_rotZ

            width_val  = float(grip_width[step])  if grip_width  is not None else np.nan
            # effort_val = float(grip_effort[step]) if grip_effort is not None else np.nan
            gap_t = width_to_gap(width_val, last_gap); last_gap = gap_t

            c_left_t, c_right_t = get_roi_centers_from_gap(
                gap=gap_t, thickness=ROI_THICKNESS, close_axis_idx=close_i,
                center_offset=ROI_CENTER_OFFSET, anchor=ANCHOR_MODE
            )

            corners_L_local = box_corners_local(c_left_t,  extent)
            corners_R_local = box_corners_local(c_right_t, extent)
            set_lineset_points_world(roi_left,  corners_L_local, T_robot_grip_vis)
            set_lineset_points_world(roi_right, corners_R_local, T_robot_grip_vis)

            if first:
                vis.add_geometry(pcd, reset_bounding_box=True)
                if SHOW_GRIP_AXIS and grip_axis is not None:
                    grip_axis.transform(T_robot_grip_vis)
                    vis.add_geometry(grip_axis, reset_bounding_box=False)
                if SHOW_EEF_FRAME and eef_axis is not None:
                    eef_axis.transform(T_robot_grip)
                    vis.add_geometry(eef_axis, reset_bounding_box=False)
                if SHOW_ROI:
                    vis.add_geometry(roi_left,  reset_bounding_box=False)
                    vis.add_geometry(roi_right, reset_bounding_box=False)

                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                ctr.set_front([0.0, 0.0, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.45)

                T_prev_grip = T_robot_grip_vis.copy()
                T_prev_eef_grip = T_robot_grip.copy()
                first = False
            else:
                vis.update_geometry(pcd)
                if SHOW_GRIP_AXIS and grip_axis is not None:
                    delta_grip = T_robot_grip_vis @ np.linalg.inv(T_prev_grip)
                    grip_axis.transform(delta_grip)
                    vis.update_geometry(grip_axis)
                if SHOW_EEF_FRAME and eef_axis is not None:
                    delta_eef_grip = T_robot_grip @ np.linalg.inv(T_prev_eef_grip)
                    eef_axis.transform(delta_eef_grip)
                    vis.update_geometry(eef_axis)
                if SHOW_ROI:
                    vis.update_geometry(roi_left)
                    vis.update_geometry(roi_right)
                T_prev_grip = T_robot_grip_vis.copy()
                T_prev_eef_grip = T_robot_grip.copy()


            # --- Depth 시각화 ---
            if SHOW_DEPTH_WIN:
                # 현재 프레임
                if SHOW_DEPTH_CUR:
                    xyz_base_cur = pc_base_cur[:, :3].astype(np.float64) if pc_base_cur.size else np.zeros((0,3))
                    xyz_cam_cur = transform_points(base_to_camera, xyz_base_cur)
                    depth_cur = project_points_to_depth(
                        xyz_cam_cur, DEPTH_W, DEPTH_H, K_depth, dist=DIST_DEPTH, assume_z_in_m=True
                    )
                    color_cur = depth_to_color(depth_cur, colormap=cv2.COLORMAP_TURBO,
                                               p_lo=5, p_hi=95, text="")
                    cv2.imshow("Depth (Now PointCloud)", color_cur)

                # 누적(fused-all, 현재 포함)
                if SHOW_DEPTH_FUSED:
                    xyz_cam_fused = transform_points(base_to_camera, xyz_base_fused)
                    depth_fused = project_points_to_depth(
                        xyz_cam_fused, DEPTH_W, DEPTH_H, K_depth, dist=DIST_DEPTH, assume_z_in_m=True
                    )
                    color_fused = depth_to_color(depth_fused, colormap=cv2.COLORMAP_TURBO,
                                                 p_lo=5, p_hi=95, text="")
                    cv2.imshow("Depth (Stack PointCloud)", color_fused)

                # 누적-현재 (메모리 전용 = fused - current, voxel 차집합)
                if SHOW_DEPTH_MEMO:
                    xyz_base_fused_only = xyz_base_fused
                    if pc_base_cur.size:
                        vs = getattr(preprocess_fused, "_temporal_voxel_size", 0.005)
                        k_fused = voxel_keys_from_xyz(xyz_base_fused_only, vs)
                        k_cur   = voxel_keys_from_xyz(pc_base_cur[:, :3].astype(np.float64), vs)
                        _, idx_fused, _ = np.intersect1d(k_fused, k_cur, return_indices=True)
                        mask_mem_only = np.ones(k_fused.shape[0], dtype=bool)
                        if idx_fused.size > 0:
                            mask_mem_only[idx_fused] = False
                        xyz_mem_only = xyz_base_fused_only[mask_mem_only]
                    else:
                        xyz_mem_only = xyz_base_fused_only

                    xyz_cam_mem = transform_points(base_to_camera, xyz_mem_only)
                    depth_mem = project_points_to_depth(
                        xyz_cam_mem, DEPTH_W, DEPTH_H, K_depth, dist=DIST_DEPTH, assume_z_in_m=True
                    )
                    color_mem = depth_to_color(depth_mem, colormap=cv2.COLORMAP_TURBO,
                                               p_lo=5, p_hi=95, text="")
                    cv2.imshow("Depth (Diff PointCloud)", color_mem)

            # ROI occupancy 로그 (Grip-local) — fused 기준
            if xyz_robot.shape[0] > 0:
                Rg = T_robot_grip_vis[:3, :3]; tg = T_robot_grip_vis[:3, 3]
                pts_local = (xyz_robot - tg) @ Rg.T
                mask_L = points_in_roi_local(pts_local, c_left_t,  extent)
                mask_R = points_in_roi_local(pts_local, c_right_t, extent)
                N_L, N_R = int(mask_L.sum()), int(mask_R.sum())
                S = min(N_L, N_R)
                print(f"[{step}/{num_steps-1}] gap={gap_t:.4f}m | N_L={N_L}, N_R={N_R}, S(min)={S}")
            else:
                print(f"[{step}/{num_steps-1}] (no fused points)")

            controller.vis_changed = False

        vis.poll_events()
        vis.update_renderer()
        if SHOW_DEPTH_WIN:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
        time.sleep(0.01)

    vis.destroy_window()
    if SHOW_DEPTH_WIN:
        if SHOW_DEPTH_CUR:   cv2.destroyWindow("Depth (Now PointCloud)")
        if SHOW_DEPTH_FUSED: cv2.destroyWindow("Depth (Stack PointCloud)")
        if SHOW_DEPTH_MEMO:  cv2.destroyWindow("Depth (Diff PointCloud)")

# ==================== I/O 래퍼 ====================
class EpisodeAnalyzer:
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