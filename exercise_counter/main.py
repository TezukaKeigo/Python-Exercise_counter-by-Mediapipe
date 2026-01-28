import os
import sys
import datetime


def save_to_log(exercise_type, count):
    """保存运动记录到日志文件"""
    log_file = "exercise_log.txt"

    # 如果日志文件不存在，创建并添加标题
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("运动记录日志\n")
            f.write("=" * 40 + "\n\n")

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 运动类型中文名
    exercise_names = {
        "squat": "深蹲",
        "pushup": "俯卧撑",
        "situp": "仰卧起坐",
        "jump_rope": "跳绳"
    }

    exercise_name = exercise_names.get(exercise_type, exercise_type)

    # 构建日志条目
    log_entry = f"[{current_time}] {exercise_name}: {count}次\n"

    # 追加到日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"本次运动已记录到 {log_file}")


def show_log():
    """显示所有运动记录"""
    log_file = "exercise_log.txt"
    if os.path.exists(log_file):
        print("\n" + "=" * 40)
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())
        print("=" * 40)
    else:
        print("暂无运动记录")


def main():
    """主函数 - 选择运动，执行对应模块"""
    print("=" * 40)
    print("运动检测系统")
    print("=" * 40)

    while True:
        print("\n选择运动类型:")
        print("1. 深蹲")
        print("2. 俯卧撑")
        print("3. 仰卧起坐")
        print("4. 跳绳")
        print("5. 查看运动记录")
        print("q. 退出")

        choice = input("\n请选择 (1/2/3/4/5/q): ").strip()

        if choice == 'q':
            print("退出程序")
            break

        if choice == '5':
            show_log()
            input("\n按回车键返回主菜单...")
            continue

        if choice not in ['1', '2', '3', '4']:
            print("无效选择，请重新输入")
            continue

        # 直接执行对应模块的完整程序
        try:
            if choice == '1':
                print("\n启动深蹲计数器...")
                import squat_counter
                if hasattr(squat_counter, 'main'):
                    # 执行模块并获取计数结果
                    count = squat_counter.main()
                    if count is not None and count > 0:
                        record = input(f"\n检测到 {count} 次深蹲，是否记录？(y/n): ").strip().lower()
                        if record == 'y':
                            save_to_log("squat", count)
                    else:
                        print("未检测到有效深蹲次数")

            elif choice == '2':
                print("\n启动俯卧撑计数器...")
                import pushup_counter
                if hasattr(pushup_counter, 'main'):
                    count = pushup_counter.main()
                    if count is not None and count > 0:
                        record = input(f"\n检测到 {count} 次俯卧撑，是否记录？(y/n): ").strip().lower()
                        if record == 'y':
                            save_to_log("pushup", count)
                    else:
                        print("未检测到有效俯卧撑次数")

            elif choice == '3':
                print("\n启动仰卧起坐计数器...")
                import situp_counter
                if hasattr(situp_counter, 'main'):
                    count = situp_counter.main()
                    if count is not None and count > 0:
                        record = input(f"\n检测到 {count} 次仰卧起坐，是否记录？(y/n): ").strip().lower()
                        if record == 'y':
                            save_to_log("situp", count)
                    else:
                        print("未检测到有效仰卧起坐次数")

            elif choice == '4':
                print("\n启动跳绳计数器...")
                import jump_rope_counter
                if hasattr(jump_rope_counter, 'main'):
                    count = jump_rope_counter.main()
                    if count is not None and count > 0:
                        record = input(f"\n检测到 {count} 次跳绳，是否记录？(y/n): ").strip().lower()
                        if record == 'y':
                            save_to_log("jump_rope", count)
                    else:
                        print("未检测到有效跳绳次数")

            else:
                print("无效选择，请重新输入")
                continue

        except ImportError as e:
            print(f"无法找到对应的模块: {e}")
            print("请确保以下文件存在:")
            print("  - squat_counter.py")
            print("  - pushup_counter.py")
            print("  - situp_counter.py")
            print("  - jump_rope_counter.py")
        except Exception as e:
            print(f"程序执行出错或用户选择退出: {e}")

        print("\n" + "=" * 40)
        print("返回主菜单")


if __name__ == "__main__":
    main()