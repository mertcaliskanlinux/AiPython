import torch
import torch.nn as nn
import torch.optim as optim
import time
from matplotData.main import Plotter


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
    X_mean = X.mean(dim=0) # Her sütunun ortalamasını al
    X_std = X.std(dim=0)  # Her sütunun standart sapmasını al
    y_mean = y.mean() # Tüm çıktıların ortalamasını al
    y_std = y.std() # Tüm çıktıların standart sapmasını al

    X_normalized = (X - X_mean) / X_std # Veriyi normalize et
    y_normalized = (y - y_mean) / y_std # Çıktıyı normalize et
    
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
        optimizer.zero_grad() # Her epoch'ta gradyanları sıfırla
        outputs = model(X)   # Modelden çıktıları al
        loss = criterion(outputs, y) # Hesaplanan kayıp
        loss.backward() # Geriye doğru gradyanları hesapla
        optimizer.step() # Ağırlıkları güncelle

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

    Plotter().plot_loss(loss_values)

    model.eval()

    with torch.no_grad():
        predictions = model(X)

    # Tahminleri eski ölçeğine geri döndürme
    predictions = predictions * y_std + y_mean

    # Tahminleri ve gerçek değerleri görselleştirme


    Plotter().plot_predictions(y, y_std,y_mean,predictions)

    for i in range(len(y)):
        print(f"Girdi: {X[i].tolist()} - Tahmin: {predictions[i].item()}")


if __name__ == '__main__':
    for i in range(2):
        print(f"Deneme {i + 1}")
        time.sleep(0.5)
        main()
