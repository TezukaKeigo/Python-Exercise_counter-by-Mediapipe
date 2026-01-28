import cv2
import mediapipe as mp
import os
import numpy as np


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


def squat_counter(video_source=0):
    # 初始化 MediaPipe 的姿态检测模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 用于绘制姿态关键点的对象
    mp_drawing = mp.solutions.drawing_utils

    # 用于计算蹲起次数的变量
    up_position = False
    down_position = False
    squat_count = 0

    # 定义关键点索引
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    # 记录站立时的髋部高度（作为基准）
    stand_hip_height = None
    # 固定的下蹲阈值（髋部下移身体高度的15%就算下蹲）
    squat_threshold_ratio = 0.15

    # 捕获视频输入
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("无法打开视频源")
        return 0

    # 设置窗口属性，确保响应键盘事件
    cv2.namedWindow('Squat Counter', cv2.WINDOW_NORMAL)

    # 存储蹲起曲线的数据
    squat_history = []
    max_history_length = 100  # 最多显示最近100帧的数据

    # 用于存储身体高度（肩到髋）
    body_height = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为RGB格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 进行姿态检测
        results = pose.process(image)

        # 将处理后的图像标记为可写
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 初始化髋膝差值
        hip_knee_diff = 0

        # 确保检测到人体姿态
        if results.pose_landmarks:
            # 绘制检测到的姿态关键点
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 获取图像高度
            h, _, _ = image.shape

            # 获取左、右髋和膝盖的y坐标
            left_hip_y = results.pose_landmarks.landmark[LEFT_HIP].y * h
            right_hip_y = results.pose_landmarks.landmark[RIGHT_HIP].y * h
            left_knee_y = results.pose_landmarks.landmark[LEFT_KNEE].y * h
            right_knee_y = results.pose_landmarks.landmark[RIGHT_KNEE].y * h

            # 获取肩膀位置
            left_shoulder_y = results.pose_landmarks.landmark[11].y * h
            right_shoulder_y = results.pose_landmarks.landmark[12].y * h

            # 计算平均值
            avg_hip_y = (left_hip_y + right_hip_y) / 2
            avg_knee_y = (left_knee_y + right_knee_y) / 2
            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

            # 计算身体高度
            if body_height is None and avg_shoulder_y > 0:
                body_height = avg_hip_y - avg_shoulder_y

            # 计算髋部和膝盖的平均差值
            hip_knee_diff = avg_hip_y - avg_knee_y

            # 放宽的下蹲判断逻辑
            if stand_hip_height is None and not down_position:
                stand_hip_height = avg_hip_y

            # 判断下蹲和站起
            if stand_hip_height is not None and body_height is not None:
                hip_drop = avg_hip_y - stand_hip_height

                if hip_drop > body_height * squat_threshold_ratio:
                    down_position = True
                    up_position = False
                elif down_position and hip_drop < body_height * 0.05:
                    up_position = True
                    down_position = False

            # 计算蹲起
            if up_position:
                squat_count += 1
                print(f"蹲起次数: {squat_count}")
                up_position = False
                stand_hip_height = avg_hip_y

        # 更新蹲起历史数据
        squat_history.append(hip_knee_diff)
        if len(squat_history) > max_history_length:
            squat_history.pop(0)

        # 绘制蹲起曲线
        if len(squat_history) > 1:
            graph_width = 300
            graph_height = 150
            graph_x = 10
            graph_y = 80

            overlay = image.copy()
            cv2.rectangle(overlay, (graph_x, graph_y),
                          (graph_x + graph_width, graph_y + graph_height),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            cv2.rectangle(image, (graph_x, graph_y),
                          (graph_x + graph_width, graph_y + graph_height),
                          (255, 255, 255), 1)

            cv2.line(image, (graph_x, graph_y + graph_height // 2),
                     (graph_x + graph_width, graph_y + graph_height // 2),
                     (200, 200, 200), 1)

            if len(squat_history) > 1:
                if max(squat_history) != min(squat_history):
                    normalized_data = [(value - min(squat_history)) /
                                       (max(squat_history) - min(squat_history))
                                       for value in squat_history]
                else:
                    normalized_data = [0.5] * len(squat_history)

                for i in range(1, len(normalized_data)):
                    x1 = int(graph_x + (i - 1) * graph_width / len(squat_history))
                    y1 = int(graph_y + graph_height - normalized_data[i - 1] * graph_height)
                    x2 = int(graph_x + i * graph_width / len(squat_history))
                    y2 = int(graph_y + graph_height - normalized_data[i] * graph_height)

                    if len(squat_history) - i < 5:
                        color = (0, 255, 0)
                    elif normalized_data[i] > 0.7:
                        color = (255, 255, 0)
                    elif normalized_data[i] < 0.3:
                        color = (255, 0, 0)
                    else:
                        color = (0, 200, 255)

                    cv2.line(image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(image, 'Squat Curve',
                        (graph_x + 5, graph_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if hip_knee_diff != 0:
                cv2.putText(image, f'Current: {hip_knee_diff:.1f}',
                            (graph_x + 5, graph_y + graph_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 显示蹲起计数
        cv2.putText(image, f'Squat Count: {squat_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 显示处理后的图像
        cv2.imshow('Squat Counter', image)

        # 键盘控制
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            print("退出程序，结束蹲起计数")
            print(f"本次共完成了{squat_count}个蹲起")
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    return squat_count


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

    # 如果在选择环节未退出，启动蹲起计数
    count = 0  # 初始化计数
    if source is not None:
        try:
            count = squat_counter(source)
            print(f"视频播放结束，共计数{count}个蹲起")
        except Exception as e:
            print(f"程序运行出错: {e}")
        finally:
            # 清理临时文件
            if isinstance(source, str) and source == "converted_video.mp4" and os.path.exists(source):
                try:
                    os.remove(source)
                except:
                    pass

    return count  # 返回计数结果


# 只有直接运行这个文件时才执行main()
if __name__ == "__main__":
    main()