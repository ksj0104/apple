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

# 전역 변수
driver = None
running = False

GRID_COLS = 19
GRID_ROWS = 11

grid = [[0 for i in range(GRID_COLS + 1)] for j in range(GRID_ROWS + 1)]
psum = [[0 for i in range(GRID_COLS + 1)] for j in range(GRID_ROWS + 1)]
grid_axis = [[0 for i in range(GRID_COLS + 1)] for j in range(GRID_ROWS + 1)]

def start_browser(url, status_label):
    global driver, running
    running = True

    # Chrome 열기
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # JavaScript 삽입 - S 키로 캔버스 저장
    js_script = """
    document.addEventListener('keydown', function(e) {
        if (e.key === 's' || e.key === 'S') {
            const startBtn = document.querySelector('.AppleGame_startButton__9xL5W');
            if (startBtn) {
                startBtn.click();
            }
    
            // ✅ 1초(1000ms) 딜레이 후 캔버스 저장 시도
            setTimeout(() => {
                const canvas = document.querySelector('.AppleGame_canvas__hyqxE');
                if (canvas) {
                    const link = document.createElement('a');
                    link.href = canvas.toDataURL("image/png");
                    link.download = 'apple.png';
                    link.click();
                } else {
                    alert("Canvas not found after delay!");
                }
            }, 150); // 1000 = 1초 (필요시 더 늘리기)
        }
    });
    """
    driver.execute_script(js_script)
    status_label.config(text="브라우저 실행됨. S 키로 저장, ESC 키로 종료 대기 중...")

    # ESC 키 감지 루프
    while running:
        if keyboard.is_pressed("esc"):
            stop_browser(status_label)
            break
        elif keyboard.is_pressed("r"):
            image_path = "C:/Users/ksj0104/Downloads/apple.png"
            temp = ocr(image_path)
            for row in range(1, GRID_ROWS + 1):
                for col in range(1, GRID_COLS + 1):
                    grid[row][col] = temp[row-1][col-1]

            # ✅ 이미지 삭제
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"{image_path} 삭제 완료")
            else:
                print("이미지 파일이 존재하지 않습니다.")
        elif keyboard.is_pressed("b"):
            calibration() # 마우스 좌표 캘리브레이션하기

        elif keyboard.is_pressed("f"):
            refresh()
            find_rect()


        time.sleep(0.1)

def use_axis(x, y):
    grid[x][y] = 0
    return


def refresh():
    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            psum[row][col] = psum[row - 1][col] + psum[row][col - 1] - psum[row - 1][col - 1] + grid[row][col]


def find_rect():
    ret = []

    for row in range(1, GRID_ROWS + 1):
        for col in range(1, GRID_COLS + 1):
            for row2 in range(row, GRID_ROWS + 1):
                for col2 in range(col, GRID_COLS + 1):
                    sum = psum[row2][col2] - psum[row2][col-1] - psum[row-1][col2] + psum[row-1][col-1]
                    if sum == 10:
                        ret.append([row, col, row2, col2])
                        break
                    elif sum > 10:
                        break
    return ret

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
    canvas_width = size['width']
    canvas_height = size['height']
    offset_y = 146

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


def show_image(path):
    # 이미지 열기
    img = Image.open(path)
    img = img.resize((300, 300))  # 적절한 크기로 조정

    # Tkinter에서 사용할 이미지로 변환
    tk_img = ImageTk.PhotoImage(img)

    # 이미지 라벨에 표시
    image_label.config(image=tk_img)
    image_label.image = tk_img  # 🔥 참조 유지 필수!



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

    # 이미지 표시용 라벨
    image_label = tk.Label(root)
    image_label.pack()

    root.mainloop()
