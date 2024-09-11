import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


# Veri seti oluşturma ve normalize etme
def generate_house_data():
    data = [
        ([50, 1], 150000),
        ([60, 2], 180000),
        ([80, 2], 220000),
        ([100, 3], 300000),
        ([150, 4], 450000),
        ([200, 5], 600000),
        ([250, 5], 750000),
        ([300, 6], 900000)
    ]
    
    X = torch.tensor([x for x, _ in data], dtype=torch.float32)
    y = torch.tensor([y for _, y in data], dtype=torch.float32).view(-1, 1)

    # Veriyi normalize etme
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    y_mean = y.mean()
    y_std = y.std()

    X_normalized = (X - X_mean) / X_std
    y_normalized = (y - y_mean) / y_std
    
    return X_normalized, y_normalized, X_mean, X_std, y_mean, y_std


# Modeli tanımlama
class SimpleHousePricePredictor(nn.Module):
    def __init__(self):
        super(SimpleHousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(2, 20)  # Daha basit bir gizli katman
        self.fc2 = nn.Linear(20, 1)  # Çıkış katmanı

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Gizli katman aktivasyonu
        x = self.fc2(x)              # Çıkış katmanı
        return x

# Eğitim fonksiyonu
def train(model, X, y, epochs, target_loss):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Küçük bir öğrenme oranı
    loss_values = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

        # Hedef kayıp değerine ulaşıldığında eğitimi durdur
        if loss.item() < target_loss:
            print(f"Eğitim durduruldu: Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
            break

    return loss_values

# Ana fonksiyon
def main():
    X, y, X_mean, X_std, y_mean, y_std = generate_house_data()

    model = SimpleHousePricePredictor()
    epochs = 500
    loss_values = train(model, X, y, epochs, target_loss=0.001)

    # Kayıp değerlerini görselleştirme
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('train_loss_plot.png')
    plt.close()

    model.eval()

    with torch.no_grad():
        predictions = model(X)

    # Tahminleri eski ölçeğine geri döndürme
    predictions = predictions * y_std + y_mean

    # Tahminleri ve gerçek değerleri görselleştirme
    plt.figure()
    plt.scatter(range(len(y)), y * y_std + y_mean, color='blue', label='Gerçek Fiyatlar')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Tahminler')
    plt.xlabel('Ev İndeksi')
    plt.ylabel('Fiyat')
    plt.title('Gerçek Ev Fiyatları ve Model Tahminleri')
    plt.legend()
    plt.grid(True)
    plt.savefig('house_price_predictions.png')
    plt.close()

    for i in range(len(y)):
        print(f"Girdi: {X[i].tolist()} - Tahmin: {predictions[i].item()}")


if __name__ == '__main__':
    for i in range(20):
        print(f"Deneme {i + 1}")
        time.sleep(0.5)
        main()
