from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import keyboard

# ✅ 브라우저 열기
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://apple-game.okdongdongdong.com/#google_vignette")  # 여기에 실제 URL 넣기

# ✅ JavaScript 삽입: 캔버스를 저장하는 함수 정의
script = """
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
        }, 100); // 1000 = 1초 (필요시 더 늘리기)
    }
});

"""
driver.execute_script(script)

print("브라우저가 열렸습니다. 웹페이지에서 'S' 키를 누르면 캔버스 이미지가 저장됩니다.")
#
# # ✅ 사용자 대기
try:
    while True:
        if keyboard.is_pressed('esc'):
            print("🛑 ESC 입력 감지! 종료합니다.")
            driver.quit()
            break
        time.sleep(0.1)
except KeyboardInterrupt:
    driver.quit()