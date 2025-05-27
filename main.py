import tkinter as tk
from threading import Thread
import keyboard
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import tkinter as tk
from PIL import Image, ImageTk
from apple import *
import pyautogui, time
from selenium.webdriver.common.by import By
from ml import *
import numpy as np
import os

# 전역 변수
driver = None
running = False

GRID_COLS = 19
GRID_ROWS = 11
offset_y = 146

grid = [[0 for i in range(GRID_COLS + 1)] for j in range(GRID_ROWS + 1)]
psum = [[0 for i2 in range(GRID_COLS + 1)] for j2 in range(GRID_ROWS + 1)]
grid_axis = [[0 for i3 in range(GRID_COLS + 1)] for j3 in range(GRID_ROWS + 1)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델 불러오기 (한 번만)
model = SimpleCNN()
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.to(device)
model.eval()

def split_image_into_grid(image, tile_size=120):
    width, height = image.size
    tiles = []

    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)

            # 120x120로 자르기 (끝부분은 남은 크기만큼 잘림)
            tile = image.crop((left, top, right, bottom))
            tiles.append(tile)

    return tiles

def ocr(image_path):
    white_threshold=245
    enhance_range = 200
    # 1. 이미지 열기 & 그레이스케일 변환
    img = Image.open(image_path).convert("L")  # 'L' = grayscale

    # ✅ 이미지 1.5배 확대
    scale = 10
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, Image.LANCZOS)  # 고품질 리사이징

    # 2. NumPy 배열로 변환
    img_np = np.array(img)

    # # 3. 흰색(밝은 값)만 유지, 나머지 0
    # mask = img_np >= white_threshold  # True for white-ish pixels
    # img_np[:] = 0  # 전체를 검정으로
    # img_np[mask] = 255  # 흰색만 유지

    # 3. 밝은 회색 → 흰색으로 밀어올림 (선 강조)
    img_np[img_np >= enhance_range] = 255

    # 4. 흰색이 아닌 나머지는 검정 처리
    mask = img_np >= white_threshold
    img_np[:] = 0
    img_np[mask] = 255

    image = Image.fromarray(img_np)
    image = image.resize((2420, 1460), Image.LANCZOS)

    # ✅ 상하좌우 65px씩 crop (패딩 제거)
    crop_margin = 70
    cropped_image = image.crop((
        crop_margin,  # left
        crop_margin,  # top
        image.width - crop_margin,  # right
        image.height - crop_margin  # bottom
    ))

    # 필요 시 덮어쓰기 또는 새로운 이미지로 저장
    image = cropped_image
    tiles = split_image_into_grid(image)

    x = 1
    y = 1
    for i, tile in enumerate(tiles):
        grid[x][y] = test(model, device, tile)
        y += 1
        if y > GRID_COLS:
            x += 1
            y = 1

    for i in range(1, GRID_ROWS + 1):
        for j in range(1, GRID_COLS + 1):
            print(grid[i][j], end=" ")
        print('')
    os.remove(image_path)

def load_image(driver):
    js_script = """
    const canvas = document.querySelector('.AppleGame_canvas__hyqxE');
    if (canvas) {
        const link = document.createElement('a');
        link.href = canvas.toDataURL("image/png");
        link.download = 'apple.png';
        link.click();
    } else {
        alert("Canvas not found after delay!");
    }
    """
    driver.execute_script(js_script)

def start_browser(url, status_label):
    global driver, running
    running = True

    # Chrome 열기
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    image_path = "C:/Users/ksj0104/Downloads/apple.png"

    if os.path.exists(image_path):
        os.remove(image_path)
    status_label.config(text="브라우저 실행됨. S 키로 저장, ESC 키로 종료 대기 중...")

    # calibration()  # 마우스 좌표 캘리브레이션하기
    # ESC 키 감지 루프
    while running:
        if keyboard.is_pressed("esc"):
            stop_browser(status_label)
            break

        elif keyboard.is_pressed("q"):
            load_image(driver)
        elif keyboard.is_pressed("w"):
            if os.path.exists(image_path):
                ocr(image_path)
                refresh()
                draw_rect(driver, find_rect())
            else:
                load_image(driver)

        time.sleep(0.1)


def refresh():
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            psum[row][col] = psum[row - 1][col] + psum[row][col - 1] - psum[row - 1][col - 1] + grid[row][col]


def find_rect():
    ret = []

    # 9, 1 찾기
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            if grid[row][col] == 9:
                for row2 in range(row, GRID_ROWS + 1):
                    for col2 in range(col, GRID_COLS + 1):
                        sum = psum[row2][col2] - psum[row2][col - 1] - psum[row - 1][col2] + psum[row - 1][col - 1]
                        if sum == 10:
                            ret.append([row, col, row2, col2])
                            break
                        elif sum > 10:
                            break
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            if grid[row][col] == 8:
                for row2 in range(row, GRID_ROWS + 1):
                    for col2 in range(col, GRID_COLS + 1):
                        sum = psum[row2][col2] - psum[row2][col - 1] - psum[row - 1][col2] + psum[row - 1][col - 1]
                        if sum == 10:
                            ret.append([row, col, row2, col2])
                            break
                        elif sum > 10:
                            break
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            if grid[row][col] == 0 or grid[row][col] == 8 or grid[row][col] == 9:
                continue
            for row2 in range(row, GRID_ROWS + 1):
                for col2 in range(col, GRID_COLS + 1):
                    sum = psum[row2][col2] - psum[row2][col-1] - psum[row-1][col2] + psum[row-1][col-1]
                    if sum == 10:
                        ret.append([row, col, row2, col2])
                        break
                    elif sum > 10:
                        break
    return ret


def draw_rect(driver, rects):
    driver.execute_script("document.querySelectorAll('.overlay-box').forEach(el => el.remove());")
    for row1, col1, row2, col2 in rects:

        def get_center_by_grid(row, col):
            return 28 + col * 48 - 24, 28 + row * 48 - 24

        # 좌상단과 우하단 중점 좌표
        x1, y1 = get_center_by_grid(row1, col1)
        x2, y2 = get_center_by_grid(row2, col2)

        # 드래그를 위한 패딩 조정
        x1 += 787
        y1 += 324
        x2 += 797
        y2 += 334
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.1)
        pyautogui.mouseUp()
        time.sleep(0.1)
        # break


def calibration():
    canvas = driver.find_element(By.CLASS_NAME, "AppleGame_canvas__hyqxE")

    PADDING = 28  # 좌우 상하 패딩
    CANVAS_WIDTH = 968
    CANVAS_HEIGHT = 584

    CELL_WIDTH = (CANVAS_WIDTH - 2 * PADDING) // GRID_COLS
    CELL_HEIGHT = (CANVAS_HEIGHT - 2 * PADDING) // GRID_ROWS

    # ✅ 브라우저 뷰포트 기준 좌표와 크기 가져오기
    location = canvas.location
    size = canvas.size

    canvas_x = location['x']
    canvas_y = location['y']
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = canvas_x + PADDING + col * CELL_WIDTH + CELL_WIDTH // 2
            y = canvas_y + PADDING + row * CELL_HEIGHT + CELL_HEIGHT // 2
            grid_axis[row + 1][col + 1] = [x, y+offset_y]
            print([x, y+offset_y], end =' ')
        print('')
    return

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

    tk.Button(root, text="브라우저 열기", command=lambda: launch(url_entry, status_label), bg="#4CAF50", fg="white").pack(
        pady=10)

    root.mainloop()
