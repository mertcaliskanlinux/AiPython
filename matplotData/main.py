import matplotlib.pyplot as plt
class Plotter():
    def __init__(self):
        self.loss_values = []

    # Kayıp değerlerini görselleştirme
    def plot_loss(self, loss_values):
        plt.figure()
        plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.savefig('train_loss_plot.png')
        plt.close()


     # Tahminleri ve gerçek değerleri görselleştirme
    def plot_predictions(self, y,y_std,y_mean, predictions):
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


