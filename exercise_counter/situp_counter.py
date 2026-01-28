import cv2
import mediapipe as mp
import math
import os


def convert_video(input_path, output_path):
    """将视频转换为MP4格式"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True


def situp_counter(video_source=0):
    """
    仰卧起坐计数主函数
    :param video_source: 摄像头ID (整数) 或视频文件路径 (字符串)
    :return: 仰卧起坐次数
    """
    # ===================== 初始化配置 =====================
    # 保存原始的choice和video_path
    choice = None
    video_path = None

    # 根据输入类型设置参数
    if isinstance(video_source, int):
        # 摄像头
        choice = '1'
    elif isinstance(video_source, str):
        # 视频文件
        if os.path.exists(video_source):
            choice = '2'
            video_path = video_source
        else:
            print(f"视频文件不存在: {video_source}")
            return 0
    else:
        print(f"无效的视频源: {video_source}")
        return 0

    # 初始化MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 核心参数（新增运行状态变量）
    count = 0  # 完成次数
    stage = None  # 当前姿态状态：down/up/None
    angle_threshold_up = 60  # 坐起角度阈值
    angle_threshold_down = 100  # 躺下角度阈值
    is_fullscreen = False  # 全屏状态标记
    is_running = True  # 运行状态：True=运行中，False=已暂停

    def calculate_angle(a, b, c):
        """计算三点构成的夹角（b为顶点），返回0-180°"""
        ax, ay = a
        bx, by = b
        cx, cy = c

        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.hypot(v1[0], v1[1])
        magnitude_v2 = math.hypot(v2[0], v2[1])

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0

        angle_rad = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    # ===================== 窗口配置 =====================
    cv2.namedWindow('situp counter', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('situp counter', 1280, 720)

    # ===================== 摄像头与主循环 =====================
    if choice == '1':
        cap = cv2.VideoCapture(0)
    elif choice == '2':
        cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            if choice == '2':
                print("视频播放结束")
                break
            else:
                print("错误：无法读取摄像头画面，请检查摄像头是否正常")
                break

        frame = cv2.flip(frame, 1)  # 水平翻转画面，镜像显示
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 姿态检测（始终执行，仅计数逻辑受运行状态控制）
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ===================== 计数逻辑 =====================
        if is_running:
            try:
                # 提取关键点
                landmarks = results.pose_landmarks.landmark
                shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])
                hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1],
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])
                knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image.shape[0])

                # 计算角度并绘制
                angle = calculate_angle(shoulder, hip, knee)
                cv2.putText(image, f"Angle: {int(angle)}",
                            (int(hip[0]) - 50, int(hip[1]) + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 计数逻辑
                if angle > angle_threshold_down:
                    stage = "down"
                if angle < angle_threshold_up and stage == "down":
                    stage = "up"
                    count += 1
                    print(f"仰卧起坐次数：{count}")

            except Exception as e:
                # 未检测到人时重置姿态状态
                stage = None
                # 提示文字
                hint_text = "no person detected,pause counting"
                # 1. 获取文字尺寸（宽w，高h）
                text_size, _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                w, h = text_size
                # 2. 计算画面中心坐标
                cx, cy = image.shape[1] // 2, image.shape[0] // 2
                # 3. 计算文字居中的左上角坐标
                text_x = cx - w // 2
                text_y = cy + h // 2
                # 4. 计算背景框坐标（向文字四周留20px边距，更美观）
                box_x1, box_y1 = text_x - 20, text_y - h - 20
                box_x2, box_y2 = text_x + w + 20, text_y + 20
                # 绘制半透明红色背景
                overlay = image.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                # 绘制白色居中文字
                cv2.putText(image, hint_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ===================== 绘制计数、姿态、运行状态 =====================
        # 计数（红色）
        cv2.putText(image, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 姿态状态（仅运行中且有状态时显示）
        if is_running and stage is not None:
            cv2.putText(image, f"Stage: {stage}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 运行状态（绿色=运行中，橙色=暂停）
        run_status = "running | press S to pause" if is_running else "paused | press S to run, R to reset"
        status_color = (0, 255, 0) if is_running else (255, 165, 0)
        cv2.putText(image, run_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # ===================== 绘制姿态关键点与连线 =====================
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3),  # 红色关键点
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)  # 白色连线
        )

        # ===================== 按键交互 =====================
        cv2.imshow('situp counter', image)
        key = cv2.waitKey(1) & 0xFF

        # 开始/暂停切换（S键）
        if key == ord('s'):
            is_running = not is_running
        # 重置（R键）
        if key == ord('r'):
            count = 0
            stage = None
            print("count reset")  # 终端也提示重置
        # 退出（Q键或ESC键）
        if key == ord('q') or key == 27:
            print("结束仰卧起坐计数")
            break

        # 检测窗口关闭事件
        try:
            if cv2.getWindowProperty('situp counter', cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

    # ===================== 释放资源 =====================
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\n本次仰卧起坐最终完成次数：{count}")

    return count


def main():
    """独立的程序入口函数"""
    # ===================== 初始化配置 =====================
    while True:
        print("\n请选择输入来源：")
        print("1. 使用摄像头")
        print("2. 上传视频文件")
        print("q. 退出程序")
        choice = input("请输入您的选择：")
        if choice == 'q':
            print("程序退出")
            return 0
        if choice == '2':
            video_path = input("请输入视频文件路径（或输入 'q' 返回上级菜单）：")
            if video_path.lower() == 'q':
                continue

            # 检查文件是否存在
            if not os.path.exists(video_path):
                print("文件不存在，请重新输入")
                continue

            # 转换为MP4格式
            converted_video_path = "converted_video.mp4"
            if convert_video(video_path, converted_video_path):
                count = situp_counter(converted_video_path)
                # 清理临时文件
                if os.path.exists(converted_video_path):
                    try:
                        os.remove(converted_video_path)
                    except:
                        pass
                return count
            else:
                print("视频转换失败，请尝试其他文件")
                continue

        if choice == '1':
            count = situp_counter(0)
            return count

        print("无效选择")


# 只有直接运行这个文件时才执行main()
if __name__ == "__main__":
    count = main()
    print(f"仰卧起坐计数结果: {count}")