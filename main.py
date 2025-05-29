import tkinter as tk
from threading import Thread
import keyboard
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import tkinter as tk
from PIL import Image, ImageTk
from apple import *
import pyautogui, time
from selenium.webdriver.common.by import By
from ml import *
from mylogic import *
import numpy as np
import os
import base64
import io

# 전역 변수
driver = None
running = False

GRID_COLS = 0
GRID_ROWS = 0
offset_y = 146

grid = []
psum = []
grid_axis = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델 불러오기 (한 번만)
model = SimpleCNN()
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.to(device)
model.eval()

target_image = "C:/Users/ksj0104/Downloads/apple_unit.png"
template = cv2.imread(target_image, cv2.IMREAD_COLOR)
w, h = template.shape[1], template.shape[0]

def split_image_into_grid(image, center_axis):
    tiles = []
    for x, y in center_axis:
        left = x - w//2
        top = y - h//2
        right = x + w//2
        bottom = y + h//2
        tile = image.crop((left, top, right, bottom))
        tiles.append(tile)
    return tiles


def load_image():
    global driver
    js_script = """
    const canvas = document.querySelector('.AppleGame_canvas__hyqxE');
    if (canvas) {
        return canvas.toDataURL("image/png").split(',')[1];  // base64만 추출
    } else {
        return null;
    }
    """
    base64_data = driver.execute_script(js_script)

    if base64_data is None:
        raise ValueError("Canvas not found.")

    # 2️⃣ base64 → bytes
    img_bytes = base64.b64decode(base64_data)

    # 3️⃣ bytes → NumPy 배열 → OpenCV 이미지
    nparr = np.frombuffer(img_bytes, np.uint8)
    opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR 포맷
    return opencv_img

def ocr(img):
    global GRID_COLS, GRID_ROWS, offset_y, device, w ,h, template, grid, grid_axis
    # 템플릿 매칭

    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # ✅ threshold 이상인 위치를 모두 가져오기
    threshold = 0.92  # 적절한 값 조절

    loc = np.where(result >= threshold)

    detected = []
    for pt in zip(*loc[::-1]):
        if all(np.linalg.norm(np.array(pt) - np.array(d)) > 10 for d in detected):
            detected.append(pt)

    center_axis = []
    for i, (x, y) in enumerate(detected):
        cx = x + x + w
        cy = y + y + h
        center_axis.append([cx // 2, cy // 2])

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

    total_apple_count = len(adjusted_center_axis)
    GRID_COLS = len(current_group)
    GRID_ROWS = total_apple_count // GRID_COLS

    grid = [[0 for i in range(GRID_COLS)] for j in range(GRID_ROWS)]
    grid_axis = [[0 for i in range(GRID_COLS)] for j in range(GRID_ROWS)]

    detect_apple_debug(adjusted_center_axis)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tiles = split_image_into_grid(pil_img, adjusted_center_axis)

    white_threshold=245
    enhance_range = 200

    for i, tile in enumerate(tiles):

        img_convert = tile.convert('L')
        scale = 10
        new_size = (int(img_convert.width * scale), int(img_convert.height * scale))
        img_resized  = img_convert.resize(new_size, Image.LANCZOS)  # 고품질 리사이징

        # 2. NumPy 배열로 변환
        img_np = np.array(img_resized)

        # 3. 밝은 회색 → 흰색으로 밀어올림 (선 강조)
        img_np[img_np >= enhance_range] = 255

        # 4. 흰색이 아닌 나머지는 검정 처리
        mask = img_np >= white_threshold
        img_np[:] = 0
        img_np[mask] = 255

        image = Image.fromarray(img_np)

        filename = f"cropped/tile_{i + 1:03d}.png"  # tile_001.png, tile_002.png, ...
        image.save(filename)
        grid[i // GRID_COLS][i % GRID_COLS] = test(model, device, image)
        print( i // GRID_COLS, i % GRID_COLS, " = " , grid[i // GRID_COLS][i % GRID_COLS])

    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            print(grid[i][j], end=" ")
        print('')

    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            print(grid_axis[i][j], end=" ")
        print('')

    solver = solve(grid, GRID_ROWS, GRID_COLS)
    for (r1, c1), (r2, c2) in solver:
        start = grid_axis[r1][c1]
        end = grid_axis[r2][c2]
        x1 = start[0] + 753 - 15
        y1 = start[1] + 335 - 15
        x2 = end[0] + 753 + 15
        y2 = end[1] + 335 + 15

        pyautogui.moveTo(x1, y1, duration=0.1)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.2)
        pyautogui.mouseUp()
        time.sleep(0.1)  # 드래그 간 간격


def start_browser(url, status_label):
    global driver, running
    running = True
    # Chrome 열기
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    status_label.config(text="브라우저 실행됨. S 키로 저장, ESC 키로 종료 대기 중...")

def detect_apple_debug(center_axis):
    global w, h, driver
    driver.execute_script("document.querySelectorAll('.overlay-box').forEach(el => el.remove());")
    idx = 0
    for x, y in center_axis:
        top_left = (x - w // 2, y - h // 2)
        grid_axis[idx // GRID_COLS][idx % GRID_COLS] = [x, y]
        idx = idx + 1
        # JS 삽입: HTML 요소로 박스 덧씌우기
        js_script = f"""
               const canvas = document.querySelector('.AppleGame_canvas__hyqxE');
               if (!canvas) return;

               const rect = document.createElement('div');
               rect.className = 'debug-rect';
               rect.style.position = 'absolute';
               rect.style.border = '2px solid red';
               rect.style.left = (canvas.offsetLeft + {top_left[0]}) + 'px';
               rect.style.top = (canvas.offsetTop + {top_left[1]}) + 'px';
               rect.style.width = '{w}px';
               rect.style.height = '{h}px';
               rect.style.pointerEvents = 'none';
               rect.style.zIndex = '9999';
               canvas.parentNode.appendChild(rect);
           """
        driver.execute_script(js_script)

def stop_browser(status_label):
    global driver, running
    running = False
    if driver:
        driver.quit()
        driver = None
    status_label.config(text="브라우저 종료됨. 프로그램 대기 중.")

def launch(url_entry, status_label):
    url = url_entry.get()
    if not url.startswith("http"):
        status_label.config(text="❌ URL 형식이 올바르지 않습니다.")
        return

    thread = Thread(target=start_browser, args=(url, status_label))
    thread.daemon = True
    thread.start()

def on_analyze_click():
    global driver
    if driver:
        image = load_image()
        ocr(image)

def on_exit_click(status_label):
    stop_browser(status_label)
    root.quit()  # Tkinter 앱 종료

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Canvas Image Saver")
    root.geometry("450x200")

    tk.Label(root, text="URL 입력:").pack(pady=5)
    url_entry = tk.Entry(root, width=50)
    url_entry.insert(0, "https://apple-game.okdongdongdong.com/#google_vignette")  # ← 기본 URL
    url_entry.pack()

    status_label = tk.Label(root, text="대기 중...", fg="blue")
    status_label.pack(pady=10)

    tk.Button(root, text="링크 열기", command=lambda: launch(url_entry, status_label), bg="#4CAF50", fg="white").pack(pady=10)
    tk.Button(root, text="분석(Q)", command=on_analyze_click, bg="blue", fg="white").pack(pady=5)
    tk.Button(root, text="종료(ESC)", command=lambda: on_exit_click(status_label), bg="red", fg="white").pack(pady=5)

    root.mainloop()
