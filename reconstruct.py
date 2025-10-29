# reconstruct.py
import os
import cv2
import numpy as np


# -------------------------
# 1) PLY 로더 (ASCII 전용)
# -------------------------
def read_ply_vertices_ascii(path):
    """
    간단한 ASCII PLY 파서 (vertex만 읽음).
    이진(binary) PLY일 경우 Open3D 등으로 대체 필요.
    return: (N,3) float64, 단위: PLY 원본과 동일(보통 미터)
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n, head_end = 0, 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            n = int(ln.split()[2])
        if ln.strip() == "end_header":
            head_end = i + 1
            break
    pts = []
    for ln in lines[head_end : head_end + n]:
        sp = ln.strip().split()
        if len(sp) >= 3:
            try:
                x, y, z = map(float, sp[:3])
                if (x != 0.0) or (y != 0.0) or (z != 0.0):
                    pts.append([x, y, z])
            except Exception:
                pass
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------
# 2) C2D 포인트클라우드 → Depth 이미지 재구성
# --------------------------------------------
def reconstruct_depth_from_ply_c2d(
    ply_file,
    depth_w,
    depth_h,
    fx_d,
    fy_d,
    cx_d,
    cy_d,
    dist=None,           # None이면 왜곡 0(권장: rectified depth)
    assume_z_in_m=True,  # PLY z가 meter면 True → mm로 변환
):
    """
    C2D로 생성된 point cloud는 Depth 카메라 좌표계라고 가정.
    Depth intrinsics로 3D→2D 투영 후 Z-buffer(가까운 점)로 rasterize.
    출력: uint16 depth(mm), shape (H,W)
    """
    # 카메라 내참 행렬
    Kd = np.array([[fx_d, 0.0,  cx_d],
                   [0.0,  fy_d, cy_d],
                   [0.0,  0.0,  1.0]], dtype=np.float64)

    # 왜곡 파라미터 (기본 rectified)
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    # 포인트 로드
    pts = read_ply_vertices_ascii(ply_file)
    if pts.size == 0:
        return np.zeros((depth_h, depth_w), dtype=np.uint16)

    # 카메라 앞쪽(Z>0)만 사용
    valid = pts[:, 2] > 0
    pts = pts[valid]
    if pts.size == 0:
        return np.zeros((depth_h, depth_w), dtype=np.uint16)

    # 3D→2D 투영 (벡터화)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, Kd, dist)
    uv = imgpts.reshape(-1, 2)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)

    # 이미지 바운더리 체크
    inb = (u >= 0) & (u < depth_w) & (v >= 0) & (v < depth_h)
    u, v, z = u[inb], v[inb], pts[inb, 2]

    # 단위 변환 (m → mm)
    if assume_z_in_m:
        z_mm = (z * 1000.0).astype(np.float64)
    else:
        z_mm = z.astype(np.float64)

    # Z-buffer (가까운 점 우선: 같은 픽셀에 여러 점 → 최소 깊이)
    depth = np.zeros((depth_h, depth_w), dtype=np.uint16)
    lin = (v * depth_w + u).astype(np.int64)

    # 같은 픽셀끼리 묶어서 최소값 구하기
    order = np.argsort(lin)
    lin_sorted = lin[order]
    z_sorted = z_mm[order]
    uniq, first = np.unique(lin_sorted, return_index=True)
    mins = np.minimum.reduceat(z_sorted, first)
    mins = np.clip(mins, 0, 65535).astype(np.uint16)
    depth.ravel()[uniq] = mins
    return depth


# --------------------------------
# 3) 보기 좋은 시각화(퍼센타일)
# --------------------------------
def depth_to_color(depth_u16, colormap=cv2.COLORMAP_TURBO, p_lo=5, p_hi=95):
    """
    depth_u16: uint16(mm)
    - 홀(0)을 제외하고 퍼센타일 [p_lo, p_hi]로 정규화
    - 보기 좋은 TURBO 컬러맵 기본
    """
    h, w = depth_u16.shape
    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    scaled = np.zeros((h, w), dtype=np.uint8)

    if np.any(valid):
        p = np.percentile(depth[valid], [p_lo, p_hi]).astype(np.float32)
        lo, hi = float(p[0]), float(p[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # fallback: 유효 범위 전체
            lo, hi = float(depth[valid].min()), float(depth[valid].max())

        # 클리핑 후 선형 스케일
        d_clip = np.clip(depth, lo, hi)
        scaled[valid] = ((d_clip[valid] - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8)
    # 홀(0)은 그대로 0 → 컬러맵에서 어두운 색/파란색 등으로 표현

    color = cv2.applyColorMap(scaled, colormap)
    return color


# -------------
# 4) 메인
# -------------
if __name__ == "__main__":
    # --- 입력 PLY 경로 ---
    here = os.path.dirname(os.path.abspath(__file__))
    ply_file = os.path.join(here, "point_clouds", "point_cloud.ply")

    if not os.path.exists(ply_file):
        raise FileNotFoundError(
            f"PLY not found: {ply_file}\n"
            f"point_clouds/point_cloud.ply 를 먼저 생성하세요."
        )

    # --- 카메라 파라미터 (사용자 제공 Depth 프로파일) ---
    DEPTH_W, DEPTH_H = 320, 288
    fx_d = 252.69204711914062
    fy_d = 252.65277099609375
    cx_d = 166.12030029296875
    cy_d = 176.21173095703125

    # (권장) rectified depth 가정 → 왜곡 0
    dist_d = None

    # (옵션) raw depth 왜곡을 OpenCV Rational 8계수로 쓰고 싶다면 아래 주석 해제
    # 제공 순서: k1,k2,k3,k4,k5,k6, p1,p2
    # OpenCV 순서: k1,k2, p1,p2, k3,k4,k5,k6
    # k1,k2,k3,k4,k5,k6 = 11.690221786499023, 5.343990802764893, 0.17299659550189972, 12.01732349395752, 9.254467010498047, 1.1656895875930786
    # p1, p2 = 6.46333719487302e-05, 1.3530498108593747e-05
    # dist_d = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

    # --- 재구성 ---
    depth_img = reconstruct_depth_from_ply_c2d(
        ply_file=ply_file,
        depth_w=DEPTH_W, depth_h=DEPTH_H,
        fx_d=fx_d, fy_d=fy_d, cx_d=cx_d, cy_d=cy_d,
        dist=dist_d,
        assume_z_in_m=True  # PLY z가 meter면 True
    )

    # 저장(선택)
    out_dir = os.path.join(here, "reconstruct_out")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth_mm.npy"), depth_img)
    cv2.imwrite(os.path.join(out_dir, "depth_mm.png"), depth_img)  # 16-bit PNG

    # 시각화
    color = depth_to_color(depth_img, colormap=cv2.COLORMAP_TURBO, p_lo=5, p_hi=95)
    cv2.imshow("Reconstructed Depth (C2D)", color)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
