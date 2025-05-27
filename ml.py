import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# ✅ 모델 정의 (학습 때와 같은 구조여야 함)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

def train():
    # ✅ 전처리 정의
    transform = transforms.Compose([
        transforms.Grayscale(),               # 흑백 변환 (1채널)
        transforms.Resize((28, 28)),          # MNIST 형식에 맞춤
        transforms.ToTensor(),                # Tensor 변환
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    # ✅ ImageFolder를 사용한 데이터셋 로딩
    dataset = datasets.ImageFolder(root="./trainSet", transform=transform)

    # ✅ DataLoader로 배치화
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()
    model.to(device)

    # ✅ 3. 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 4. 학습 루프
    for epoch in range(5):  # 5 에폭만 예시로
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "digit_cnn.pth")
    print("✅ 모델 저장 완료: digit_cnn.pth")



def evaluation():

    # ✅ 전처리 정의
    transform = transforms.Compose([
        transforms.Grayscale(),               # 흑백 변환 (1채널)
        transforms.Resize((28, 28)),          # MNIST 형식에 맞춤
        transforms.ToTensor(),                # Tensor 변환
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
    model.to(device)
    model.eval()  # 평가 모드 전환

    # ✅ 동일한 전처리 + 테스트셋 구성
    test_dataset = datasets.ImageFolder(root="./trainSet", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # ✅ 테스트 루프
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(labels, predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"✅ 테스트 정확도: {100 * correct / total:.2f}%")

def test(model, device, image):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return pred

if __name__ == '__main__':
    train()
    evaluation()


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SimpleCNN()
    # model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
    # model.to(device)
    # model.eval()  # 평가 모드 전환
    #
    # test(model, "./cropped/tile_1.png")
