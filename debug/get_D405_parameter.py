#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense D405: 현재 카메라 설정값/메타데이터 읽기 (수정본)
- depth/color 스트림 시작 → 워밍업 → 현재 옵션값과 범위, 프레임 메타데이터,
  intrinsics/extrinsics, depth_units 등을 출력
- --json 으로 저장 가능
"""
import pyrealsense2 as rs
import json
from datetime import datetime
import argparse

def enum_name(e):
    s = str(e)
    return s.split('.')[-1]

def safe_iter_metadata_keys():
    # 표준: Python Enum은 직접 순회 가능
    try:
        return list(rs.frame_metadata_value)
    except Exception:
        # 폴백: 대문자 멤버만 추출
        return [getattr(rs.frame_metadata_value, n)
                for n in dir(rs.frame_metadata_value) if n.isupper()]

def dump_sensor_options(sensor: rs.sensor):
    out = {"sensor_name": sensor.get_info(rs.camera_info.name)}
    try:
        opts = list(sensor.get_supported_options())  # 권장: 지원 옵션만
    except Exception:
        # 폴백: 모든 enum 멤버 순회 후 supports()로 거르기
        try:
            opts = list(rs.option)
        except Exception:
            opts = [getattr(rs.option, n) for n in dir(rs.option) if n.isupper()]

    for opt in opts:
        try:
            if sensor.supports(opt):
                rng = sensor.get_option_range(opt)  # (min, max, step, default)
                try:
                    val = sensor.get_option(opt)
                except Exception:
                    val = None
                out[enum_name(opt)] = {
                    "value": val,
                    "min": rng.min, "max": rng.max,
                    "step": rng.step, "default": rng.default
                }
        except Exception:
            continue
    return out

def dump_intrinsics(profile: rs.video_stream_profile):
    intr = profile.get_intrinsics()
    return {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy,
        "ppx": intr.ppx, "ppy": intr.ppy,
        "model": str(intr.model),                # 가독성 위해 문자열화
        "coeffs": list(intr.coeffs)
    }

def dump_frame_metadata(frame: rs.frame):
    out = {}
    for key in safe_iter_metadata_keys():
        try:
            if frame.supports_frame_metadata(key):
                out[enum_name(key)] = frame.get_frame_metadata(key)
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--json", type=str, default=None, help="저장 경로 (예: d405_current_settings.json)")
    ap.add_argument("--warmup", type=int, default=30, help="워밍업 프레임 수")
    ap.add_argument("--serial", type=str, default=None, help="특정 시리얼 장치 사용")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipe.start(cfg)

    try:
        # 워밍업(자동 노출/게인 안정화)
        for _ in range(max(0, args.warmup)):
            pipe.wait_for_frames()

        # 한 세트 프레임
        frames = pipe.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError("프레임 획득 실패")

        dev = profile.get_device()
        sensors = list(dev.query_sensors())

        result = {
            "timestamp": datetime.now().isoformat(),
            "device_name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "options_by_sensor": [],
            "depth_units_m_per_unit": None,
            "intrinsics": {},
            "extrinsics_depth_to_color": {},
            "frame_metadata": {"depth": {}, "color": {}}
        }

        # 센서 옵션값들
        for s in sensors:
            result["options_by_sensor"].append(dump_sensor_options(s))

        # depth_units (m/unit)
        try:
            depth_sensor = dev.first_depth_sensor()
            if depth_sensor.supports(rs.option.depth_units):
                result["depth_units_m_per_unit"] = depth_sensor.get_option(rs.option.depth_units)
        except Exception:
            pass

        # Intrinsics
        dprof = depth.profile.as_video_stream_profile()
        cprof = color.profile.as_video_stream_profile()
        result["intrinsics"]["depth"] = dump_intrinsics(dprof)
        result["intrinsics"]["color"] = dump_intrinsics(cprof)

        # Extrinsics (depth -> color)
        try:
            ext = dprof.get_extrinsics_to(cprof)
            result["extrinsics_depth_to_color"] = {
                "rotation_3x3_rowmajor": list(ext.rotation),
                "translation_xyz_m": list(ext.translation)
            }
        except Exception:
            pass

        # 프레임 메타데이터
        result["frame_metadata"]["depth"] = dump_frame_metadata(depth)
        result["frame_metadata"]["color"] = dump_frame_metadata(color)

        # 출력
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 파일 저장 옵션
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 저장 완료: {args.json}")

    finally:
        pipe.stop()

if __name__ == "__main__":
    main()
