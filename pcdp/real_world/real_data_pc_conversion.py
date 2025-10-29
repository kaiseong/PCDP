# real_data_pc_conversion.py
from typing import Sequence, Tuple, Dict, Optional, Union, List
import os
import pathlib
import cv2
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm
import open3d as o3d
import torch
import pcdp.common.mono_time as mono_time
try:
    import pytorch3d.ops as torch3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. FPS will use fallback method.")
from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from pcdp.common.RISE_transformation import xyz_rot_transform

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

class PointCloudPreprocessor:
    """Pointcloud preprocessing with optional temporal memory for fused cache export.
    When enable_temporal=True and export_mode='fused', each call to `process(points)`
    updates a stateful voxel map (per episode) with exponential decay and returns
    an Nx7 array: [x,y,z,r,g,b,c], where `c` is temporal confidence (recency).
    """

    # ----------[ NEW ]: Orbbec→OpenCV 변환 유틸 ----------
    @staticmethod
    def _is_orbbec_intrinsics(obj) -> bool:
        return all(hasattr(obj, a) for a in ("fx", "fy", "cx", "cy"))

    @staticmethod
    def _is_orbbec_distortion(obj) -> bool:
        return all(hasattr(obj, a) for a in ("k1","k2","k3","k4","k5","k6","p1","p2"))

    @staticmethod
    def _as_cv_K_from_orbbec_intrinsics(orbbec_intr):
        # OBCameraIntrinsic → OpenCV K
        K = np.array([[float(orbbec_intr.fx), 0.0, float(orbbec_intr.cx)],
                      [0.0, float(orbbec_intr.fy), float(orbbec_intr.cy)],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    @staticmethod
    def _as_cv_dist_from_orbbec_distortion(orbbec_dist):
        # Orbbec(k1..k6, p1, p2) → OpenCV rational 8계수 [k1,k2,p1,p2,k3,k4,k5,k6]
        k1 = float(orbbec_dist.k1); k2 = float(orbbec_dist.k2); k3 = float(orbbec_dist.k3)
        k4 = float(orbbec_dist.k4); k5 = float(orbbec_dist.k5); k6 = float(orbbec_dist.k6)
        p1 = float(orbbec_dist.p1); p2 = float(orbbec_dist.p2)
        return np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

    @staticmethod
    def convert_orbbec_depth_params(depth_intrinsics, depth_distortion):
        """외부에서 편히 쓰는 변환기: (Orbbec intr, dist) -> (K(3x3), dist(8,))."""
        K = PointCloudPreprocessor._as_cv_K_from_orbbec_intrinsics(depth_intrinsics)
        dist = PointCloudPreprocessor._as_cv_dist_from_orbbec_distortion(depth_distortion)
        return K, dist
    # -----------------------------------------------------

    def __init__(self, 
                enable_sampling=False,
                target_num_points=1024,
                enable_transform=True,
                extrinsics_matrix=None,
                enable_cropping=True,
                workspace_bounds=None,
                enable_filter=False,
                nb_points=10,
                sor_std=1.7,
                use_cuda=True,
                verbose=False,
                enable_temporal=False,
                export_mode='off',
                temporal_voxel_size=0.005,
                temporal_decay=0.90,
                temporal_c_min=0.20,
                # hmm...
                temporal_prune_every: int=1,
                stable_export: bool = False,
                enable_occlusion_prune: bool = True,
                depth_width: Optional[int] = 320,
                depth_height: Optional[int] = 288,
                K_depth: Optional[Sequence[Sequence[float]]] = None,    # 3x3
                dist_depth: Optional[Sequence[float]] = None,           # None or 5/8계수
                erode_k: int = 1,
                z_unit: str = 'm',   # 'm' or 'mm',
                occl_patch_radius: int = 2
                ):

        if extrinsics_matrix is None:
            self.extrinsics_matrix = np.array([
                [  0.007131,  -0.91491,    0.403594,  0.05116],
                [ -0.994138,   0.003833,   0.02656,  -0.00918],
                [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
                [  0.,         0. ,        0. ,        1.      ]
            ])
        else:
            self.extrinsics_matrix = np.array(extrinsics_matrix)
            
        # Default workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = [
                [-0.000, 0.715],    # X range (m)
                [-0.400, 0.350],    # Y range (m)
                [-0.100, 0.400]     # z range
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
        if K_depth is None:
            self._K_depth = np.array([
                [252.69204711914062, 0.0, 166.12030029296875],
                [0.0, 252.65277099609375, 176.21173095703125],
                [0.0, 0.0, 1.0]
                ], dtype=np.float64)
        else:
            if self._is_orbbec_intrinsics(K_depth):
                self._K_depth = self._as_cv_K_from_orbbec_intrinsics(K_depth)
            else:
                self._K_depth = np.array(K_depth, dtype=np.float64)
        assert self._K_depth.shape == (3, 3), f"K_depth must be 3x3, got {self._K_depth.shape}"

        if dist_depth is None:
            # [k1 k2 p1 p2 k3 k4 k5 k6]
            self._dist_depth = np.array(
            [11.690222, 5.343991, 0.000065, 0.000014, 0.172997, 12.017323, 9.254467, 1.165690], dtype=np.float64)
        else:
            if self._is_orbbec_distortion(dist_depth):
                self._dist_depth = self._as_cv_dist_from_orbbec_distortion(dist_depth)
            else:
                self._dist_depth = np.asarray(dist_depth, dtype=np.float64)
                if self._dist_depth.size not in (4, 5, 8):
                    raise ValueError(f"dist_depth must have 5 or 8 coeffs, got {self._dist_depth.size}")



        self.target_num_points = target_num_points
        self.nb_points = nb_points
        self.sor_std = sor_std
        self.enable_transform = enable_transform
        self.enable_cropping = enable_cropping
        self.enable_sampling = enable_sampling
        self.enable_filter = enable_filter
        self.use_cuda = use_cuda and torch.cuda.is_available() and PYTORCH3D_AVAILABLE
        self.verbose = verbose
        
        # temporal memory
        self.enable_temporal = enable_temporal
        self.export_mode = export_mode
        self._temporal_voxel_size = float(temporal_voxel_size)
        self._temporal_decay = float(temporal_decay)
        self._temporal_c_min = float(temporal_c_min)
        
        self._prune_every = int(max(1, temporal_prune_every))
        self._stable_export = bool(stable_export)
            
        # occlusion prune 설정
        self.enable_occlusion_prune = bool(enable_occlusion_prune)
        self._depth_w = int(depth_width) 
        self._depth_h = int(depth_height) 
        self._erode_k = int(erode_k)
        self._z_unit = str(z_unit)
        self.occl_patch_radius = int(occl_patch_radius)

        # extrinsics_matrix는 Camera->Base 이므로 Base->Camera를 미리 준비
        self._base_to_cam = None
        if self.enable_occlusion_prune:
            if self.extrinsics_matrix is None:
                raise ValueError("enable_occlusion_prune=True인데 extrinsics_matrix가 없습니다.")
            self._base_to_cam = np.linalg.inv(np.array(self.extrinsics_matrix, dtype=np.float64))
            if self._K_depth is None or self._depth_w is None or self._depth_h is None:
                raise ValueError("occlusion prune에는 depth_width/height와 K_depth가 필요합니다.")

        self._frame_idx = 0
        self._mem_keys = np.empty((0,), dtype=np.dtype((np.void, 12)))
        self._mem_xyz = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb = np.empty((0, 3), dtype=np.float32)
        self._mem_step = np.empty((0, ), dtype=np.int32)

        if 0.0 < self._temporal_decay < 1.0 and 0.0 < self._temporal_c_min < 1.0:
            self._max_age_steps = int(np.floor(np.log(self._temporal_c_min)/np.log(self._temporal_decay)))
        else:
            self._max_age_steps = 0

        if self.verbose:
            print(f"PointCloudPreprocessor initialized:")
            print(f"  - Transform: {self.enable_transform}")
            print(f"  - Cropping: {self.enable_cropping}")
            print(f"  - Sampling: {self.enable_sampling} (target: {self.target_num_points})")
            print(f"  - CUDA: {self.use_cuda}")
            
    def __call__(self, points):
        """Process pointcloud through the full pipeline."""
        return self.process(points)
    
    def _project_points_cam(self, xyz_cam: np.ndarray):
        """카메라 좌표계 3D -> (u,v,z, inb_mask). z는 z_unit에 맞춘 float."""
        
        if xyz_cam.size == 0:
            return (np.array([], np.int32),)*2 + (np.array([], np.float64), np.array([], bool))
        valid = xyz_cam[:, 2] > 0.0
        pts = xyz_cam[valid]
        if pts.size == 0:
            return (np.array([], np.int32),)*2 + (np.array([], np.float64), np.zeros(0, bool))
        rvec = np.zeros((3,1), np.float64); tvec = np.zeros((3,1), np.float64)
        dist = np.zeros(5, np.float64) if self._dist_depth is None else self._dist_depth
        imgpts, _ = cv2.projectPoints(pts.reshape(-1,1,3), rvec, tvec, self._K_depth, dist)
        uv = imgpts.reshape(-1,2)
        u = np.rint(uv[:,0]).astype(np.int32)
        v = np.rint(uv[:,1]).astype(np.int32)
        inb = (u >= 0) & (u < self._depth_w) & (v >= 0) & (v < self._depth_h)
        return u[inb], v[inb], pts[inb, 2].astype(np.float64), valid.nonzero()[0][inb]

    def _rasterize_min(self, u: np.ndarray, v: np.ndarray, z: np.ndarray):
        """Z-buffer(min)로 깊이 이미지(uint16, mm) 생성."""
        depth = np.zeros((self._depth_h, self._depth_w), dtype=np.uint16)
        if u.size == 0:
            return depth
        z_mm = (z * 1000.0) if self._z_unit == 'm' else z
        lin = (v * self._depth_w + u).astype(np.int64)
        order = np.argsort(lin)
        lin_s, z_s = lin[order], z_mm[order]
        uniq, first = np.unique(lin_s, return_index=True)
        mins = np.minimum.reduceat(z_s, first)
        depth.ravel()[uniq] = np.clip(mins, 0, 65535).astype(np.uint16)
        return depth

    def _erode_min(self, depth_u16: np.ndarray):
        """깊이가 작은(가까운) 값을 국소적으로 확장시키는 최소필터."""
        k = self._erode_k
        if k <= 0: 
            return depth_u16
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k+1, 2*k+1))
        d = depth_u16.copy()
        zero = (d == 0)
        d[zero] = 65535
        d = cv2.erode(d, kernel)
        d[zero] = 0
        return d

    def _occlusion_prune_memory(self, now_xyz_base: np.ndarray):
        """현재 프레임 기반 Z_now로 메모리(_mem_*)에서 '앞에 있는' 과거 점 삭제."""
        if (not self.enable_occlusion_prune) or self._mem_keys.size == 0:
            return
        if now_xyz_base.size == 0:
            return
        # 1) now(Base->Cam) → Z_now
        Nn = now_xyz_base.shape[0]
        xyz_now_cam = (self._base_to_cam @ np.c_[now_xyz_base[:,:3], np.ones((Nn,1))].T).T[:, :3]
        u_n, v_n, z_n, _ = self._project_points_cam(xyz_now_cam)
        Z_now = self._rasterize_min(u_n, v_n, z_n)
        Z_now = self._erode_min(Z_now)  # 노이즈 완화

        # 2) mem(Base->Cam) 픽셀 비교
        Nm = self._mem_xyz.shape[0]
        xyz_mem_cam = (self._base_to_cam @ np.c_[self._mem_xyz, np.ones((Nm,1))].T).T[:, :3]
        u_m, v_m, z_m, map_idx = self._project_points_cam(xyz_mem_cam)
        if u_m.size == 0:
            return

        z_mem_mm = (z_m * 1000.0) if self._z_unit == 'm' else z_m
        z_now_at = self._patch_min(Z_now, u_m, v_m, self.occl_patch_radius).astype(np.float64)

        # 삭제 규칙: now 유효 & (mem이 충분히 앞) → 삭제
        del_local = (z_now_at > 0) & (z_mem_mm < z_now_at)
        
        if not np.any(del_local):
            return

        keep_global = np.ones(self._mem_xyz.shape[0], dtype=bool)
        keep_global[map_idx[del_local]] = False

        # 실제 삭제
        self._mem_keys = self._mem_keys[keep_global]
        self._mem_xyz  = self._mem_xyz[keep_global]
        self._mem_rgb  = self._mem_rgb[keep_global]
        self._mem_step = self._mem_step[keep_global]

        if hasattr(self, "_mem_seen"):
            self._mem_seen = self._mem_seen[keep_global]
        if hasattr(self, "_mem_miss"):
            self._mem_miss = self._mem_miss[keep_global]


    def reset_temporal(self):
        """Call at the start of each episode."""
        self._frame_idx = 0
        self._mem_keys = np.empty((0, ), dtype=np.dtype((np.void, 12)))
        self._mem_xyz   = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb   = np.empty((0, 3), dtype=np.float32)
        self._mem_step  = np.empty((0,),  dtype=np.int32)

    def voxel_keys_from_xyz(self, xyz:np.ndarray):
        grid = np.floor(xyz / self._temporal_voxel_size).astype(np.int32, copy=False) # (N, 3) int32
        grid = np.ascontiguousarray(grid)
        keys = grid.view(np.dtype((np.void, 12))).ravel() # (N, ) 12byte key
        return keys
    
    def _frame_unique(self, xyz: np.ndarray, rgb: np.ndarray, last_wins: bool=False):
        if xyz.shape[0] == 0:
            return xyz, rgb
        keys = self.voxel_keys_from_xyz(xyz)
        if last_wins:
            # 뒤집어서 unique → 다시 뒤집어 인덱스 복원
            rev = keys[::-1]
            _, idx_rev = np.unique(rev, return_index=True)
            uniq_idx = (len(keys) - 1) - idx_rev
        else:
            _, uniq_idx = np.unique(keys, return_index=True)
        return xyz[uniq_idx], rgb[uniq_idx]
    
    def _merge_into_mem(self, xyz_new: np.ndarray, rgb_new: np.ndarray, now_step: int):
        if xyz_new.shape[0] == 0:
            return
        keys_new = self.voxel_keys_from_xyz(xyz_new)
        
        if self._mem_keys.size == 0:
            self._mem_keys = keys_new.copy()
            self._mem_xyz  = xyz_new.astype(np.float32, copy=False).copy()
            self._mem_rgb  = rgb_new.astype(np.float32, copy=False).copy()
            self._mem_step = np.full((xyz_new.shape[0],), now_step, dtype=np.int32)
            return
        
        common, idx_mem, idx_new = np.intersect1d(self._mem_keys, keys_new, return_indices=True)
        if common.size > 0:
            self._mem_xyz[idx_mem]  = xyz_new[idx_new]
            self._mem_rgb[idx_mem]  = rgb_new[idx_new]
            self._mem_step[idx_mem] = now_step
        
        mask_new_only = np.ones(keys_new.shape[0], dtype=bool)
        if common.size > 0:
            mask_new_only[idx_new] = False
        if mask_new_only.any():
            self._mem_keys = np.concatenate([ self._mem_keys, keys_new[mask_new_only] ], axis=0)
            self._mem_xyz  = np.concatenate([ self._mem_xyz,  xyz_new[mask_new_only] ], axis=0)
            self._mem_rgb  = np.concatenate([ self._mem_rgb,  rgb_new[mask_new_only] ], axis=0)
            self._mem_step = np.concatenate([ self._mem_step,
                                              np.full((mask_new_only.sum(),), now_step, dtype=np.int32) ], axis=0)
    
    def _prune_mem(self, now_step: int):
        if self._mem_keys.size == 0:
            return
        if (now_step % self._prune_every) != 0:
            return
        age = now_step - self._mem_step
        keep = (age <= self._max_age_steps)
        if not np.all(keep):
            self._mem_keys = self._mem_keys[keep]
            self._mem_xyz  = self._mem_xyz[keep]
            self._mem_rgb  = self._mem_rgb[keep]
            self._mem_step = self._mem_step[keep]

    def _export_array_from_mem(self, now_step: int) -> np.ndarray:
        N = self._mem_keys.size
        if N == 0:
            return np.zeros((0,7), dtype=np.float32)
        age = (now_step - self._mem_step).astype(np.float32)
        c   = (self._temporal_decay** age).astype(np.float32, copy=False)  # (N,)

        # stable order 원하면 키 기준 정렬
        if self._stable_export:
            order = np.argsort(self._mem_keys)  # 바이트키 정렬(결정적)
            xyz = self._mem_xyz[order]
            rgb = self._mem_rgb[order]
            c   = c[order]
        else:
            xyz = self._mem_xyz
            rgb = self._mem_rgb

        out = np.concatenate([xyz, rgb, c[:,None]], axis=1).astype(np.float32, copy=False)

        return out

    def _patch_min(self, Z: np.ndarray, u: np.ndarray, v: np.ndarray, r: int):
        H, W = Z.shape
        out = np.zeros_like(u, dtype=np.float64)
        for i in range(u.size):
            u0 = max(0, u[i]-r); u1 = min(W-1, u[i]+r)
            v0 = max(0, v[i]-r); v1 = min(H-1, v[i]+r)
            patch = Z[v0:v1+1, u0:u1+1]
            nz = patch[patch>0]
            out[i] = float(nz.min()) if nz.size else 0.0
        return out
        
    def process(self, points):
        """
        Process pointcloud through transformation, cropping, and sampling.
        
        Args:
            points: numpy array of shape (N, 6) containing XYZRGB data
            
        Returns:
            processed_points: numpy array of processed pointcloud
        """
        if points is None or len(points) == 0:
            if self.enable_temporal and self.export_mode == 'fused':
                self._frame_idx += 1
                return np.zeros((0,7), dtype=np.float32)
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        # Ensure points is float32
        points = points.astype(np.float32)
        t0 = mono_time.now_ms()
        # Coordinate transformation
        if self.enable_transform:
            points = self._apply_transform(points)
        # Workspace cropping
        t1 = mono_time.now_ms()
        if self.enable_cropping:
            points = self._crop_workspace(points)
        t2 = mono_time.now_ms()
        if self.enable_filter:
            points = self._apply_filter(points)
        if not self.enable_temporal or self.export_mode!="fused":
            # Point FPS sampling
            if self.enable_sampling:
                points = self._sample_points(points)
            return points
        
        t3 = mono_time.now_ms()

        now_step = self._frame_idx

        # 1) 프레임 내 voxel-unique (동일 voxel 중복 제거 – 기능 동일)
        xyz_now = points[:, :3]
        rgb_now = points[:, 3:6]

        if self.enable_occlusion_prune:
            self._occlusion_prune_memory(xyz_now)

        xyz_now, rgb_now = self._frame_unique(xyz_now, rgb_now)

        # 2) 메모리 병합: 겹치면 '현재'로 갱신 (latest-wins = c 큰 값 유지와 동일 의미)
        self._merge_into_mem(xyz_now, rgb_now, now_step)

        # 3) 주기적 프루닝(decay^age < c_min) – 기능 동일, 벡터화
        self._prune_mem(now_step)

        # 4) export: lazy decay로 c 계산해 Nx7 반환 (기능 동일)
        out = self._export_array_from_mem(now_step)
        
        # (선택) 시각화 로그: 과거일수록 페이드
        if self.verbose and out.shape[0] > 0:
            out_dbg = out.copy()
            rgb = out_dbg[:, 3:6].astype(np.float32, copy=False)
            c   = np.clip(out[:, 6:7], 0.0, 1.0)
            s = 255.0 if (rgb.max() > 1.0 + 1e-6) else 1.0
            out_dbg[:, 3:6] = np.clip((rgb/s) * c, 0.0, 1.0) * s
            print(f"time0: {t1 - t0}")
            print(f"time1: {t2 - t1}")
            print(f"time2: {t3 - t2}")
            print(f"time3: {mono_time.now_ms() - t3}")
        
        
        self._frame_idx += 1
        return out


    def _apply_transform(self, points):
        """Apply extrinsics transformation and scaling."""
        # Scale from mm to m (Orbbec specific)

        points = points[points[:, 2] > 0.0]
        point_xyz = points[:, :3] * 0.001
        
        # Apply extrinsics transformation
        point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
        point_transformed = np.dot(point_homogeneous, self.extrinsics_matrix.T)
        
        # Update XYZ coordinates
        points[:, :3] = point_transformed[:, :3]
        
        if self.verbose and len(points) > 0:
            print(f"After transform: {len(points)} points, "
                  f"XYZ range: [{points[:, :3].min(axis=0)} - {points[:, :3].max(axis=0)}]")
        
        return points
        
    def _crop_workspace(self, points):
        """Crop points to workspace bounds."""
        if len(points) == 0:
            return points
            
        mask = (
            (points[:, 0] >= self.workspace_bounds[0][0]) & 
            (points[:, 0] <= self.workspace_bounds[0][1]) &
            (points[:, 1] >= self.workspace_bounds[1][0]) & 
            (points[:, 1] <= self.workspace_bounds[1][1]) &
            (points[:, 2] >= self.workspace_bounds[2][0]) & 
            (points[:, 2] <= self.workspace_bounds[2][1])
        )
        
        cropped_points = points[mask]
        
        if self.verbose:
            print(f"After cropping: {len(cropped_points)}/{len(points)} points remain")
            
        return cropped_points

    def _apply_filter(self, points):
        if len(points) == 0:
            raise ValueError("points empty")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_points, std_ratio=self.sor_std)
        return points[ind]
        
    def _sample_points(self, points):
        """Apply farthest point sampling to reduce number of points."""
        if len(points) == 0:
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        if len(points) <= self.target_num_points:
            # Pad with zeros if not enough points
            padded_points = np.zeros((self.target_num_points, 6), dtype=np.float32)
            padded_points[:len(points)] = points
            return padded_points
            
        try:
            # Use farthest point sampling
            points_xyz = points[:, :3]
            sampled_xyz, sample_indices = self._farthest_point_sampling(
                points_xyz, self.target_num_points)
                
            # Reconstruct full points with RGB
            if self.use_cuda:
                sample_indices = sample_indices.cpu().numpy().flatten()
            else:
                sample_indices = sample_indices.numpy().flatten()
                
            sampled_points = points[sample_indices]
            
            if self.verbose:
                print(f"After sampling: {len(sampled_points)} points")
                
            return sampled_points
            
        except Exception as e:
            if self.verbose:
                print(f"FPS failed: {e}, using random sampling")
            # Fallback to random sampling
            indices = np.random.choice(len(points), self.target_num_points, replace=False)
            return points[indices]
            
    def _farthest_point_sampling(self, points, num_points):
        """Apply farthest point sampling using pytorch3d."""
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d not available")
            
        points_tensor = torch.from_numpy(points)
        
        if self.use_cuda:
            points_tensor = points_tensor.cuda()
            
        # pytorch3d expects batch dimension
        points_batch = points_tensor.unsqueeze(0)
        
        # Apply FPS
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_batch, K=[num_points])
            
        # Remove batch dimension
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        
        if self.use_cuda:
            sampled_points = sampled_points.cpu()
            
        return sampled_points.numpy(), indices

class LowDimPreprocessor:
    def __init__(self,
                 robot_to_base=None
                ):
        if robot_to_base is None:
            self.robot_to_base=np.array([
                [1., 0., 0., -0.04],
                [0., 1., 0., -0.29],
                [0., 0., 1., -0.03],
                [0., 0., 0.,  1.0]
            ])
        else:
            self.robot_to_base = np.array(robot_to_base, dtype=np.float32)
        
    
    def TF_process(self, robot_7ds):
        assert robot_7ds.shape[-1] == 7, f"robot_7ds data shape shoud be (..., 7), but got {robot_7ds.shape}"
        processed_robot7d = []
        for robot_7d in robot_7ds:
            pose_6d = robot_7d[:6]
            gripper = robot_7d[6]
            
            translation = pose_6d[:3]
            rotation = pose_6d[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
            T_k_matrix = self.robot_to_base @ eef_to_robot_base_k
            transformed_pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='ZYX'
            )
            new_robot_7d = np.concatenate([transformed_pose_6d, [gripper]])
            processed_robot7d.append(new_robot_7d)
        
        return np.array(processed_robot7d, dtype=np.float32)




def create_default_preprocessor(target_num_points=1024, use_cuda=True, verbose=False):
    """Create a preprocessor with default settings."""
    return PointCloudPreprocessor(
        target_num_points=target_num_points,
        use_cuda=use_cuda,
        verbose=verbose
    )


def downsample_obs_data(obs_data, downsample_factor=3):
    """
    Downsample observation data by taking every Nth sample.
    
    Args:
        obs_data: Dictionary of observation arrays
        downsample_factor: Factor to downsample by (e.g., 3 for 30Hz->10Hz)
        
    Returns:
        downsampled_data: Dictionary with downsampled arrays
    """
    downsampled_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            downsampled_data[key] = value[::downsample_factor].copy()
        else:
            downsampled_data[key] = value
    return downsampled_data


def align_obs_action_data(obs_data, action_data, obs_timestamps, action_timestamps):
    """
    Align observation and action data based on timestamps.
    For each obs timestamp, find the first action timestamp that comes after it.
    
    Args:
        obs_data: Dictionary of observation arrays
        action_data: Dictionary of action arrays  
        obs_timestamps: Array of observation timestamps
        action_timestamps: Array of action timestamps
        
    Returns:
        aligned_obs_data: Dictionary of aligned observation data
        aligned_action_data: Dictionary of aligned action data
        valid_indices: Indices where alignment was successful
    """
    valid_indices = []
    aligned_action_indices = []
    
    for i, obs_ts in enumerate(obs_timestamps):
        # Find first action timestamp >= obs timestamp
        future_actions = action_timestamps >= obs_ts
        if np.any(future_actions):
            action_idx = np.where(future_actions)[0][0]
            valid_indices.append(i)
            aligned_action_indices.append(action_idx)
    
    if len(valid_indices) == 0:
        print("Warning: No valid obs-action alignments found!")
        return {}, {}, []
    
    # Filter obs data to only valid indices
    aligned_obs_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_obs_data[key] = value[valid_indices]
        else:
            aligned_obs_data[key] = value
            
    # Filter action data to aligned indices
    aligned_action_data = {}
    for key, value in action_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_action_data[key] = value[aligned_action_indices]
        else:
            aligned_action_data[key] = value
    # print(f"Aligned {len(valid_indices)} obs-action pairs from {len(obs_timestamps)} obs and {len(action_timestamps)} actions")
    return aligned_obs_data, aligned_action_data, valid_indices



def process_single_episode(episode_path, pc_preprocessor=None, lowdim_preprocessor=None, downsample_factor=3):
    """
    Process a single episode: load, downsample, align, and optionally preprocess.
    
    Args:
        episode_path: Path to episode directory
        pc_preprocessor: Optional PointCloudPreprocessor instance
        lowdim_preprocessor: Optional LowDimPreprocessor instance
        downsample_factor: Factor for downsampling obs data
        
    Returns:
        episode_data: Dictionary containing processed episode data
    """

    episode_path = pathlib.Path(episode_path)

    if pc_preprocessor is not None and hasattr(pc_preprocessor, "reset_temporal"):
        pc_preprocessor.reset_temporal()
    
    obs_zarr_path = episode_path / 'obs_replay_buffer.zarr'
    action_zarr_path = episode_path / 'action_replay_buffer.zarr'
    
    if not obs_zarr_path.exists() or not action_zarr_path.exists():
        raise FileNotFoundError(f"Missing zarr files in {episode_path}")
    
    obs_replay_buffer = ReplayBuffer.create_from_path(str(obs_zarr_path), mode='r')
    action_replay_buffer = ReplayBuffer.create_from_path(str(action_zarr_path), mode='r')

    obs_data ={}
    for key in obs_replay_buffer.keys():
        obs_data[key] = obs_replay_buffer[key][:]
    
    action_data ={}
    for key in action_replay_buffer.keys():
        action_data[key] = action_replay_buffer[key][:]

    # Downsample obs data from 30Hz to 10Hz
    downsampled_obs = downsample_obs_data(obs_data, downsample_factor)
    downsampled_obs_timestamps = downsampled_obs['align_timestamp']
    action_timestamps = action_data['timestamp']
    
    # Align obs and action data based on timestamps
    aligned_obs, aligned_action, valid_indices = align_obs_action_data(
        downsampled_obs, action_data, 
        downsampled_obs_timestamps, action_timestamps)
    
    
    if len(valid_indices) == 0:
        return None
        
    # Apply pointcloud preprocessing if provided
    if pc_preprocessor is not None and 'pointcloud' in aligned_obs:
        processed_pointclouds = []
        for pc in aligned_obs['pointcloud']:
            processed_pc = pc_preprocessor.process(pc)
            processed_pointclouds.append(processed_pc)
        aligned_obs['pointcloud'] = np.array(processed_pointclouds, dtype=object)
    
    
    # Create robot_obs by concatenating pose and gripper width
    robot_eef_pose = aligned_obs['robot_eef_pose']
    robot_gripper_width = aligned_obs['robot_gripper'][:, :1] # Keep it as a column vector
    aligned_obs['robot_obs'] = np.concatenate([robot_eef_pose, robot_gripper_width], axis=1) 
    

    # TF to based on origin frames
    if lowdim_preprocessor is not None:
        aligned_obs['robot_obs'] = lowdim_preprocessor.TF_process(aligned_obs['robot_obs'])
        aligned_action['action'] = lowdim_preprocessor.TF_process(aligned_action['action'])
    
    # Combine obs and action data
    episode_data = {}
    episode_data.update(aligned_obs)
    episode_data.update(aligned_action)
    
    return episode_data


def parse_shape_meta(shape_meta: dict) -> Tuple[List[str], List[str], dict, dict]:
    """
    Parse shape_meta to extract pointcloud keys, lowdim keys, and their configurations.
    
    Args:
        shape_meta: Shape metadata dictionary from config
        
    Returns:
        pointcloud_keys: List of pointcloud observation keys
        lowdim_keys: List of lowdim observation keys  
        pointcloud_configs: Configuration for each pointcloud key
        lowdim_configs: Configuration for each lowdim key
    """
    pointcloud_keys = []
    lowdim_keys = []
    pointcloud_configs = {}
    lowdim_configs = {}
    
    # Parse obs shape meta
    obs_shape_meta = shape_meta.get('obs', {})
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape', []))
        
        if obs_type == 'pointcloud':
            pointcloud_keys.append(key)
            pointcloud_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
    
    return pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs


def validate_episode_data_with_shape_meta(episode_data: dict, shape_meta: dict) -> bool:
    """
    Validate that episode data matches the expected shape_meta.
    
    Args:
        episode_data: Processed episode data
        shape_meta: Expected shape metadata
        
    Returns:
        bool: True if validation passes
    """
    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    # Validate pointcloud data
    for key in pointcloud_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = pointcloud_configs[key]['shape']
            if len(data.shape) >= 2:
                # Check that last dimensions match expected shape
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected pointcloud key '{key}' not found in episode data")
            return False
    
    # Validate lowdim data
    for key in lowdim_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = lowdim_configs[key]['shape']
            if len(expected_shape)==1:
                if expected_shape[0] == 1 and len(data.shape) == 1:
                    continue

            if len(data.shape) >= 1:
                # Check that last dimensions match expected shape
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected lowdim key '{key}' not found in episode data")
            return False
    
    # Validate action data
    action_shape_meta = shape_meta.get('action', {})
    if 'action' in episode_data and 'shape' in action_shape_meta:
        expected_action_shape = tuple(action_shape_meta['shape'])
        actual_action_shape = episode_data['action'].shape[-len(expected_action_shape):]
        if actual_action_shape != expected_action_shape:
            print(f"Warning: Action shape mismatch. Expected: {expected_action_shape}, Got: {actual_action_shape}")
            return False
    
    return True


def _get_replay_buffer(
        dataset_path: str,
        shape_meta: dict,
        store: Optional[zarr.ABSStore] = None,
        pc_preprocessor: Optional[PointCloudPreprocessor] = None,
        lowdim_preprocessor: Optional[LowDimPreprocessor] = None,
        downsample_factor: int = 3,
        max_episodes: Optional[int] = None,
        n_workers: int = 1
) -> ReplayBuffer:
    """
    Convert Piper demonstration data to ReplayBuffer format.
    
    Args:
        dataset_path: Path to recorder_data directory containing episode folders
        shape_meta: Dictionary defining observation and action shapes/types
        store: Zarr store for output (if None, uses MemoryStore)
        pc_preprocessor: Optional pointcloud preprocessor
        lowdim_preprocessor: Optional lowdim preprocessor
        downsample_factor: Factor for downsampling obs data (30Hz -> 10Hz = 3)
        max_episodes: Maximum number of episodes to process
        n_workers: Number of worker processes (currently unused)
        
    Returns:
        replay_buffer: ReplayBuffer containing processed data
    """
    if store is None:
        store = zarr.MemoryStore()
        
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Parse shape_meta to understand data structure
    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    print(f"Parsed shape_meta:")
    print(f"  - Pointcloud keys: {pointcloud_keys}")
    print(f"  - Lowdim keys: {lowdim_keys}")
    print(f"  - Action shape: {shape_meta.get('action', {}).get('shape', 'undefined')}")
    
    # Find all episode directories
    episode_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and item.name.startswith('episode_'):
            episode_dirs.append(item)
            
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
        
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if len(episode_dirs) == 0:
        raise ValueError("No episode directories found")
    
    # Create ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    
    # Process episodes
    with tqdm(total=len(episode_dirs), desc="Processing episodes", mininterval=1.0) as pbar:
        for episode_dir in episode_dirs:
            try:
                episode_data = process_single_episode(
                    episode_dir, pc_preprocessor, lowdim_preprocessor, downsample_factor)
                
                if episode_data is not None:
                    # Validate episode data against shape_meta
                    if validate_episode_data_with_shape_meta(episode_data, shape_meta):
                        # Ensure all data are numpy arrays before adding to buffer
                        for key in episode_data.keys():
                            if isinstance(episode_data[key], list):
                                episode_data[key] = np.asarray(episode_data[key])
                        
                        # Add episode to replay buffer
                        replay_buffer.add_episode(episode_data,
                            object_codecs={'pointcloud': numcodecs.Pickle()})
                        pbar.set_postfix(
                            episodes=replay_buffer.n_episodes,
                            steps=replay_buffer.n_steps
                        )
                    else:
                        print(f"Skipping episode {episode_dir.name} due to shape validation failure")
                else:
                    print(f"Skipping empty episode: {episode_dir.name}")
                    
            except Exception as e:
                print(f"Error processing {episode_dir.name}: {e}")
                continue
                
            pbar.update(1)
    
    print(f"Successfully processed {replay_buffer.n_episodes} episodes "
        f"with {replay_buffer.n_steps} total steps")
    
    return replay_buffer