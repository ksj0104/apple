import numpy as np
from ml import *
import os


# 숫자 예측 함수
def predict_tile(tile, model, device):
    # 전처리 (모델 학습 시 사용한 것과 동일하게!)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(tile).unsqueeze(0).to(device)  # 배치 차원 추가
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred + 1  # 클래스가 0~8이면 1~9로 매핑

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 모델 불러오기 (한 번만)
    model = SimpleCNN()
    model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    cols = 19
    rows = 11
    grid = [[0 for j in range(cols)] for i in range(rows)]
    for i, tile in enumerate(tiles):
        tile.save(f"./cropped/tile_{i:03}.png")


    x = 0
    y = 0
    img_path = os.listdir("./cropped")
    for img in img_path:
        if img.endswith(".png"):
            grid[x][y] = test(model, "./cropped/" + img)
            y += 1
            if y == cols:
                x += 1
                y = 0

    for i in range(rows):
        for j in range(cols):
            print(grid[i][j], end=' ')
        print('')

    return grid






if __name__ == "__main__":

    img_path = './apple.png'
    ocr(img_path)