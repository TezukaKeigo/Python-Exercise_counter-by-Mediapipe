import sys
import time
import statistics
import os

try:
    # 在 Windows 下确保终端输出为 UTF-8，避免中文乱码
    if os.name == 'nt':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
except Exception:
    pass

try:
    import cv2
    import mediapipe as mp
except Exception as e:
    print("缺少依赖：", e)
    print("请先安装依赖：")
    print("    pip install opencv-python mediapipe")
    raise

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calibrate_baseline(pose, cap, frames=30):
    """采集若干帧，计算站立时髋部 y 的基线值（normalized）"""
    vals = []
    count = 0
    while count < frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            vals.append((lh + rh) / 2.0)
            count += 1
    if vals:
        return statistics.mean(vals)
    return None


def smooth(vals, window=5):
    if not vals:
        return None
    if len(vals) < window:
        return statistics.mean(vals)
    return statistics.mean(vals[-window:])


def jump_rope_counter(video_source=0):
    """跳绳计数主函数，返回计数结果"""
    cap = None
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(str(video_source))

    if not cap.isOpened():
        print('无法打开视频源：', video_source)
        return 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # 区分摄像头和视频文件的处理逻辑
        if isinstance(video_source, int):
            # 摄像头：不立即校准，等待自动或手动校准
            baseline = None
            is_camera = True
        else:
            # 视频文件：提前校准基线
            baseline = calibrate_baseline(pose, cap, frames=30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开头
            is_camera = False
            print(f"视频文件基线校准完成: {baseline}")

        # 实时计数状态
        jump_count = 0
        hip_vals = []
        in_jump = False
        threshold = 0.02
        smooth_window = 3

        # 摄像头专用：校准相关变量
        if is_camera:
            calibration_frames = []  # 用于校准的帧数据
            is_calibrating = True  # 是否处于校准模式
            calibration_start_time = None
            calibration_duration = 5  # 校准持续时间（秒）

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 绘制骨骼关键点
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=2)
                )

                lm = results.pose_landmarks.landmark

                targets = [
                    ('L_shoulder', mp_pose.PoseLandmark.LEFT_SHOULDER),
                    ('R_shoulder', mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    ('L_hip', mp_pose.PoseLandmark.LEFT_HIP),
                    ('R_hip', mp_pose.PoseLandmark.RIGHT_HIP),
                ]

                for name, landmark in targets:
                    idx = landmark.value
                    point = lm[idx]
                    # 规范化坐标转换为像素坐标
                    cx = int(point.x * w)
                    cy = int(point.y * h)
                    # 画圆点并标注名称和归一化坐标
                    color = (0, 255, 0) if 'shoulder' in name else (255, 128, 0)
                    cv2.circle(frame, (cx, cy), 8, color, -1)
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)  # 白色边框
                    cv2.putText(frame, f'{name}({point.x:.2f},{point.y:.2f})', (cx + 12, cy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 计算髋部归一化 y
                lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
                rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                hip_y = (lh + rh) / 2.0
                hip_vals.append(hip_y)
                if len(hip_vals) > 50:
                    hip_vals.pop(0)
                    # 平滑处理
                hip_s = smooth(hip_vals, smooth_window)
                # 摄像头自动校准
                if is_camera:
                    if is_calibrating:
                        # 收集校准数据
                        calibration_frames.append(hip_y)

                        # 如果刚开始校准，记录开始时间
                        if calibration_start_time is None:
                            calibration_start_time = time.time()

                        # 计算校准已进行的时间
                        calibration_elapsed = time.time() - calibration_start_time

                        # 显示校准信息
                        calibration_progress = min(1.0, calibration_elapsed / calibration_duration)
                        bar_width = 200
                        bar_filled = int(bar_width * calibration_progress)

                        cv2.putText(frame, 'CALIBRATING...', (w // 2 - 100, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(frame, 'Please stand still in starting position',
                                    (w // 2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # 进度条背景
                        cv2.rectangle(frame, (w // 2 - bar_width // 2, 100),
                                      (w // 2 + bar_width // 2, 110), (100, 100, 100), -1)
                        # 进度条前景
                        cv2.rectangle(frame, (w // 2 - bar_width // 2, 100),
                                      (w // 2 - bar_width // 2 + bar_filled, 110), (0, 255, 255), -1)

                        # 显示时间
                        time_left = max(0, calibration_duration - calibration_elapsed)
                        cv2.putText(frame, f'Time left: {time_left:.1f}s',
                                    (w // 2 - 60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # 如果校准时间足够，完成校准
                        if calibration_elapsed >= calibration_duration:
                            if len(calibration_frames) >= 30:
                                # 使用最后30帧计算基线
                                baseline = statistics.mean(calibration_frames[-30:])
                                is_calibrating = False
                                print(f"自动校准完成！基线值: {baseline:.4f}")
                            else:
                                # 如果没有足够的数据，使用所有数据
                                baseline = statistics.mean(calibration_frames) if calibration_frames else hip_y
                                is_calibrating = False
                                print(f"自动校准完成（使用{len(calibration_frames)}帧）！基线值: {baseline:.4f}")
                    else:
                        # 正常计数模式
                        if hip_s is not None and baseline is not None:
                            # 简单阈值检测跳跃
                            diff = baseline - hip_s

                            if diff > threshold and not in_jump:  # 髋部明显低于基线
                                in_jump = True
                            elif diff < threshold / 2 and in_jump:  # 髋部回到基线附近
                                jump_count += 1
                                in_jump = False
                else:
                    # 视频文件模式：
                    if hip_s is not None and baseline is not None:
                        # 简单阈值检测跳跃
                        diff = baseline - hip_s

                        if diff > threshold and not in_jump:  # 髋部明显低于基线
                            in_jump = True
                        elif diff < threshold / 2 and in_jump:  # 髋部回到基线附近
                            jump_count += 1
                            in_jump = False

                # 在画面上显示实时计数
                cv2.putText(frame, f'Count: {jump_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 200, 255), 2)

                # 显示基线信息
                if baseline is not None and hip_s is not None and not (is_camera and is_calibrating):
                    diff_text = f"Diff: {diff:.3f}" if 'diff' in locals() else "Diff: N/A"
                    cv2.putText(frame, f'Base: {baseline:.3f}', (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.putText(frame, f'Curr: {hip_s:.3f}', (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.putText(frame, diff_text, (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                    # 显示状态
                    status = "JUMPING" if in_jump else "GROUND"
                    status_color = (0, 0, 255) if in_jump else (0, 255, 0)
                    cv2.putText(frame, f'Status: {status}', (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    # 摄像头模式下显示提示
                    if is_camera:
                        cv2.putText(frame, 'Press C to recalibrate', (w - 250, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                # 没有检测到关键点时显示提示
                cv2.putText(frame, 'No pose detected', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Jump Rope Counter', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # 重置计数器
                jump_count = 0
                print("计数器已重置")
            elif key == ord('c'):  # 手动重新校准（仅摄像头模式）
                if is_camera:
                    is_calibrating = True
                    calibration_frames = []
                    calibration_start_time = None
                    baseline = None
                    jump_count = 0  # 重置计数器
                    in_jump = False
                    print("重新校准...")
                else:
                    print("视频文件模式下不支持重新校准")

    cap.release()
    cv2.destroyAllWindows()

    # 输出结果
    if is_camera:
        print(f'摄像头使用结束，跳绳计数: {jump_count} 次')
    else:
        print(f'视频播放结束，跳绳计数: {jump_count} 次')

    return jump_count  # 返回计数结果


def main():
    """独立运行时的入口函数"""
    while True:
        print('\n请选择输入来源：')
        print('1. 使用摄像头')
        print('2. 上传视频文件（输入文件路径）')
        print("q. 退出程序")
        choice = input('请输入您的选择: ').strip()
        if choice == '1':
            print('启动摄像头 ...')
            print('请注意：程序启动后将自动进行5秒校准，请站在起始位置保持不动。')
            print('如需重新校准，请按"C"键。')
            count = jump_rope_counter(0)
            print(f"跳绳计数完成，共{count}次")
            break
        elif choice == '2':
            path = input('请输入视频文件路径（或输入 `q` 返回上级菜单）: ').strip()
            if path.lower() == 'q':
                # 返回上级菜单
                continue
            if not os.path.exists(path):
                print('文件不存在：', path)
                continue
            print('注意：视频文件将自动使用前30帧进行校准。')
            count = jump_rope_counter(path)
            print(f"跳绳计数完成，共{count}次")
            break
        elif choice.lower() == 'q':
            print('退出')
            return 0
        else:
            print('无效选择')

    return count  # 返回计数结果


if __name__ == '__main__':
    try:
        count = main()
        print(f"最终跳绳计数: {count}")
    except KeyboardInterrupt:
        print('\n已中断，退出')