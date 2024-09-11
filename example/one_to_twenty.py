import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Veri seti oluşturma
def generate_data(num_samples):
    X = torch.arange(1, num_samples + 1, dtype=torch.float32).view(-1, 1)  # 1'den 10'a kadar
    y = X  # Basit doğrusal ilişki: y = X
    return X, y

# Modeli tanımlama
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 30)  # İlk gizli katman (1 giriş, 30 çıkış)
        self.fc2 = nn.Linear(30, 30)  # İkinci gizli katman (30 giriş, 30 çıkış)
        self.fc3 = nn.Linear(30, 1)   # Çıkış katmanı (30 giriş, 1 çıkış)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Aktivasyon fonksiyonu: ReLU
        x = torch.relu(self.fc2(x))  # Aktivasyon fonksiyonu: ReLU
        x = self.fc3(x)              # Çıkış katmanı
        return x

# Eğitim ve test verileri
X_train, y_train = generate_data(20)  # 1'den 10'a kadar olan eğitim verisi
X_test, y_test = generate_data(20)  # Test için de aynı verileri kullanıyoruz

# Model, loss ve optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Ortalama Kare Hatası (Mean Squared Error) kaybı
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim fonksiyonu
def train(model, X_train, y_train, epochs, target_loss=0.01):
    model.train()
    loss_values = []  # Eğitim sırasında kayıp değerlerini saklamak için liste

    for epoch in range(epochs):
        optimizer.zero_grad()  # Gradyanları sıfırla
        outputs = model(X_train)  # Modelin tahminleri
        loss = criterion(outputs, y_train)  # Kayıp hesapla
        loss.backward()  # Geri yayılım (backpropagation)
        optimizer.step()  # Ağırlıkları güncelle

        loss_values.append(loss.item())  # Kayıp değerini listeye ekle

        # Kayıp değeri istenen seviyeye ulaştıysa eğitimi durdur
        if loss.item() < target_loss:
            print(f"Eğitim durduruldu: Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
            break

        # Her 10 epoch'ta bir kayıp değerini yazdır
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    return loss_values

# Eğitim
epochs = 1000  # Maksimum epoch sayısı
loss_values = train(model, X_train, y_train, epochs, target_loss=0.01)

# Kayıp değerlerini görselleştirme
plt.figure()
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.savefig('train_loss_plot.png')
plt.close()  # Grafiği kapat, bellek sızıntısını önle

# Model ağırlıklarını kaydetme
torch.save(model.state_dict(), 'model_weights.pth')

# Modeli yeniden oluşturma ve ağırlıkları yükleme
model = SimpleNN()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# Modeli değerlendirme moduna alma
model.eval()

# 1'den 10'a kadar olan tahminleri yapma
with torch.no_grad():
    predictions = model(X_test)

# Tahminleri ve gerçek değerleri görselleştirme
plt.figure()
plt.scatter(X_test, y_test, color='blue', label='Gerçek Veriler')
plt.scatter(X_test, predictions, color='red', label='Tahminler')
plt.xlabel('Girdi')
plt.ylabel('Çıktı')
plt.title('Gerçek Veriler ve Model Tahminleri')
plt.legend()
plt.grid(True)
plt.savefig('predictions_vs_real.png')
plt.close()  # Grafiği kapat, bellek sızıntısını önle

# Modelin tahmin sonuçlarını yazdırma
for i in range(20):
    print(f"Girdi: {X_test[i].item()} - Tahmin: {predictions[i].item()}")







