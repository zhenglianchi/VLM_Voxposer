import cv2

# 使用 legacy 模块来创建 KCF 跟踪器
tracker = cv2.legacy.TrackerKCF_create()

# 打开视频文件或摄像头
cap = cv2.VideoCapture(0)  # 0为默认摄像头，或者可以替换为视频文件路径

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    cap.release()
    exit()

# 选择目标区域，手动选择目标框
bbox = cv2.selectROI("选择目标", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# 初始化KCF跟踪器
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新KCF跟踪器
    ret, bbox = tracker.update(frame)

    # 绘制跟踪框
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "跟踪失败", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("KCF 跟踪", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
