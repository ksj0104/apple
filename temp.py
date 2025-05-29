import cv2
import numpy as np


image_path = "C:/Users/ksj0104/Downloads/game.png"
target_image = "C:/Users/ksj0104/Downloads/apple_unit.png"
# 화면에서 이미지 위치 찾기


# 큰 이미지 (전체)
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 찾고 싶은 작은 이미지 (템플릿)
template = cv2.imread(target_image, cv2.IMREAD_COLOR)
w, h = template.shape[1], template.shape[0]

# 템플릿 매칭
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# ✅ threshold 이상인 위치를 모두 가져오기
threshold = 0.92  # 적절한 값 조절

loc = np.where(result >= threshold)


detected = []
for pt in zip(*loc[::-1]):
    if all(np.linalg.norm(np.array(pt) - np.array(d)) > 10 for d in detected):
        detected.append(pt)

grid_axis = []
center_axis = []
for i, (x, y) in enumerate(detected):
    cx = x + x + w
    cy = y + y + h
    grid_axis.append([x, y, x+w, y + h])
    center_axis.append([cx//2, cy//2])


grouped = []
current_group = [center_axis[0]]

for i in range(1, len(center_axis)):
    if abs(center_axis[i][1] - current_group[-1][1]) <= 5:
        current_group.append(center_axis[i])
    else:
        grouped.append(current_group)
        current_group = [center_axis[i]]

# 마지막 그룹 추가
if current_group:
    grouped.append(current_group)

# 평균 y 값으로 통일
adjusted_center_axis = []
for group in grouped:
    avg_y = int(np.mean([pt[1] for pt in group]))
    for pt in group:
        adjusted_center_axis.append([pt[0], avg_y])

print("중점 개수 : ", len(adjusted_center_axis))
print("한 줄 개수 : ", len(current_group))
print(w, h)
for idx, (x, y) in enumerate(adjusted_center_axis):
    top_left = (x - w // 2, y - h // 2)
    bottom_right = (x + w // 2, y + h // 2)

    # 사각형 그리기
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # 텍스트 쓰기 (번호 붙이기 등)
    text_position = (x - 10, y + 5)  # 텍스트 위치 조절 가능
    cv2.putText(img, str(idx + 1), text_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2)
#
# 결과 시각화
cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()