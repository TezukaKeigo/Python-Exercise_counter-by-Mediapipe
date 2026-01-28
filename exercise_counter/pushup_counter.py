import cv2
import mediapipe as mp
import numpy as np
from collections import deque
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


class PushupCounter:
    """俯卧撑计数器类"""

    def __init__(self):
        # 初始化MediaPipe姿态检测模型和计数器状态
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # 创建姿态检测模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 初始化计数器状态
        self.pushup_count = 0  # 初始化计数
        self.in_down_position = False  # 是否在最低点
        self.angle_history = deque(maxlen=5)  # 肘部角度历史记录
        self.stage = "up"  # 当前阶段，"up" 或 "down"

    def calculate_angle(self, a, b, c):
        """计算三点之间的角度"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle)

        return angle_deg

    def calculate_body_angle(self, shoulder, hip, ankle):
        """计算身体角度"""
        shoulder = np.array(shoulder)
        hip = np.array(hip)
        ankle = np.array(ankle)

        shoulder_hip = hip - shoulder
        hip_ankle = ankle - hip

        cosine_angle = np.dot(shoulder_hip, hip_ankle) / (np.linalg.norm(shoulder_hip) * np.linalg.norm(hip_ankle))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle)

        return angle_deg

    def process_frame(self, frame):
        """处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        h, w, c = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 提取关键点并转换为像素坐标
            left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            right_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
            left_elbow = [landmarks[13].x * w, landmarks[13].y * h]
            right_elbow = [landmarks[14].x * w, landmarks[14].y * h]
            left_wrist = [landmarks[15].x * w, landmarks[15].y * h]
            right_wrist = [landmarks[16].x * w, landmarks[16].y * h]
            left_hip = [landmarks[23].x * w, landmarks[23].y * h]
            right_hip = [landmarks[24].x * w, landmarks[24].y * h]
            left_ankle = [landmarks[27].x * w, landmarks[27].y * h]
            right_ankle = [landmarks[28].x * w, landmarks[28].y * h]

            # 计算左右手肘角度
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            avg_elbow_angle = (left_angle + right_angle) / 2

            # 计算身体角度
            body_angle = self.calculate_body_angle(right_shoulder, right_hip, right_ankle)

            # 更新角度历史
            self.angle_history.append(avg_elbow_angle)

            # 俯卧撑计数逻辑
            if avg_elbow_angle < 100 and body_angle < 15:
                if self.stage == "up" and not self.in_down_position:
                    self.stage = "down"
                    self.in_down_position = True

            elif avg_elbow_angle > 150 and self.in_down_position:
                self.pushup_count += 1
                self.stage = "up"
                self.in_down_position = False
                print(f"俯卧撑次数: {self.pushup_count}")

            # 绘制关键点和连接
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # 显示角度数值
            cv2.putText(frame, f"{int(left_angle)}",
                        (int(left_elbow[0]) + 10, int(left_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"{int(right_angle)}",
                        (int(right_elbow[0]) + 10, int(right_elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 显示信息
            info_text = f"Pushups: {self.pushup_count}"
            stage_text = f"Stage: {self.stage}"
            angle_text = f"Elbow Angle: {avg_elbow_angle:.1f}"
            body_angle_text = f"Body Angle: {body_angle:.1f}"

            cv2.putText(frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)
            cv2.putText(frame, stage_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.putText(frame, angle_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
            cv2.putText(frame, body_angle_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

        else:
            cv2.putText(frame, "No person detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def process_webcam(self):
        """使用摄像头进行实时计数"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("无法打开摄像头")
            return 0

        print("开始摄像头计数（按 'q' 退出）")
        print("按 'r' 重置计数器")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))
            frame = self.process_frame(frame)
            cv2.imshow('Pushup Counter', frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.pushup_count = 0
                self.in_down_position = False
                self.stage = "up"
                print("计数器已重置")

        cap.release()
        cv2.destroyAllWindows()
        print(f"最终俯卧撑计数: {self.pushup_count}")
        return self.pushup_count

    def process_video(self, video_path):
        """视频文件处理"""
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return 0

        print(f"处理视频: {video_path}")
        print("正在实时显示计数过程...")
        print("按 'q' 可以随时中断")

        # 显示窗口设置
        display_width = 800
        display_height = 600
        window_name = "Pushup Counter"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (display_width, display_height))
            frame = self.process_frame(frame)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户中断了处理")
                break

        cap.release()
        cv2.destroyWindow(window_name)
        print(f"最终俯卧撑计数: {self.pushup_count}")
        return self.pushup_count


def pushup_counter(video_source=0):
    """
    俯卧撑计数函数 - 保持与深蹲函数相同的接口
    :param video_source: 摄像头ID (整数) 或视频文件路径 (字符串)
    :return: 俯卧撑次数
    """
    counter = PushupCounter()

    if isinstance(video_source, int):  # 摄像头
        return counter.process_webcam()
    else:  # 视频文件
        return counter.process_video(video_source)


def main():
    """独立的程序入口函数"""
    source = None
    while True:
        print("\n请选择输入来源:")
        print("1. 使用摄像头")
        print("2. 上传视频文件")
        print("q. 退出程序")
        choice = input("请输入您的选择: ").strip()

        if choice.lower() == 'q':
            print("程序退出")
            return 0
        elif choice == '1':
            source = 0
            break
        elif choice == '2':
            while True:
                video_path = input("请输入视频文件路径 (或输入 'q' 返回上级菜单): ").strip()
                if video_path.lower() == 'q':
                    break
                elif os.path.isfile(video_path):
                    test_cap = cv2.VideoCapture(video_path)
                    if not test_cap.isOpened():
                        print("无法打开该视频文件，请检查文件格式")
                        test_cap.release()
                        continue
                    test_cap.release()

                    converted_video_path = "converted_video.mp4"
                    if convert_video(video_path, converted_video_path):
                        source = converted_video_path
                        break
                    else:
                        print("视频转换失败，请尝试其他文件")
                else:
                    print("无效的视频文件路径，请重新输入！")
            if source is not None:
                break
        else:
            print("无效输入，请重新输入！")

    # 如果在选择环节未退出，启动俯卧撑计数
    count = 0
    if source is not None:
        try:
            count = pushup_counter(source)
            print(f"视频播放结束，共计数{count}个俯卧撑")
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            # 清理临时文件
            if isinstance(source, str) and source == "converted_video.mp4" and os.path.exists(source):
                try:
                    os.remove(source)
                except:
                    pass

    return count


# 只有直接运行这个文件时才执行main()
if __name__ == "__main__":
    main()