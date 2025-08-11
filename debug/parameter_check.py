import pyorbbecsdk as ob

def main():
    # 파이프라인 및 구성 객체 초기화
    pipeline = ob.Pipeline()
    config = ob.Config()

    try:
        # 컬러 센서 프로파일 가져오기 및 활성화
        color_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        # 원하는 해상도 지정 (예: 1280x720)
        color_profile = color_profile_list.get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
        if color_profile is None:
            print("Color stream profile not found!")
            # fallback to default profile
            color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            if color_profile is None:
                print("No color stream profile found!")
                return
        config.enable_stream(color_profile)
        print(f"Color Profile: {color_profile}")

        # Depth 센서 프로파일 가져오기 및 활성화
        depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        # 원하는 해상도 지정 (예: 320x288)
        depth_profile = depth_profile_list.get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
        if depth_profile is None:
            print("Depth stream profile not found!")
            # fallback to default profile
            depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
            if depth_profile is None:
                print("No depth stream profile found!")
                return
        config.enable_stream(depth_profile)
        print(f"Depth Profile: {depth_profile}")

    except Exception as e:
        print(e)
        return

    # 파이프라인 시작
    pipeline.start(config)

    # 스트림 프로파일에서 카메라 파라미터 가져오기
    # VideoStreamProfile 객체로 캐스팅해야 get_intrinsic() 등의 함수에 접근 가능
    color_video_profile = color_profile.as_video_stream_profile()
    depth_video_profile = depth_profile.as_video_stream_profile()

    color_intrinsics = color_video_profile.get_intrinsic()
    color_distortion = color_video_profile.get_distortion()
    depth_intrinsics = depth_video_profile.get_intrinsic()
    depth_distortion = depth_video_profile.get_distortion()

    # Depth에서 Color로의 외부 파라미터(Extrinsics) 가져오기
    try:
        depth_to_color_extrinsic = depth_video_profile.get_extrinsic_to(color_profile)
        color_to_depth_extrinsic = color_profile.get_extrinsic_to(depth_profile)
    except ob.OBError as e:
        print(f"Failed to get extrinsic: {e}")
        extrinsic = None

    # 결과 출력
    print("\n--- Camera Parameters ---")
    print("\n[Color Camera]")
    print(f"  Intrinsics (fx, fy, cx, cy): {color_intrinsics.fx}, {color_intrinsics.fy}, {color_intrinsics.cx}, {color_intrinsics.cy}")
    print(f"  Distortion (k1-k6): {color_distortion.k1}, {color_distortion.k2}, {color_distortion.k3}, {color_distortion.k4}, {color_distortion.k5}, {color_distortion.k6}")
    print(f"  Distortion (p1, p2): {color_distortion.p1}, {color_distortion.p2}")


    print("\n[Depth Camera]")
    print(f"  Intrinsics (fx, fy, cx, cy): {depth_intrinsics.fx}, {depth_intrinsics.fy}, {depth_intrinsics.cx}, {depth_intrinsics.cy}")
    print(f"  Distortion (k1-k6): {depth_distortion.k1}, {depth_distortion.k2}, {depth_distortion.k3}, {depth_distortion.k4}, {depth_distortion.k5}, {depth_distortion.k6}")
    print(f"  Distortion (p1, p2): {depth_distortion.p1}, {depth_distortion.p2}")

    if color_to_depth_extrinsic:
        print("\n[Extrinsics (Depth to Color)]")
        # 수정된 속성 이름 사용
        print(f" color_to_depth_extrinsic: {color_to_depth_extrinsic}")
        print(f" depth_to_color_extrinsic: {depth_to_color_extrinsic}")
    print("-------------------------\n")


    # 파이프라인 중지
    pipeline.stop()


if __name__ == "__main__":
    main()

