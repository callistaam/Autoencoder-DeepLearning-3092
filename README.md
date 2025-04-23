# Autoencoder-DeepLearning-3092
## 🏗️ Rekonstruksi Gambar Bangunan dari Peta Segmentasi menggunakan Autoencoder

Proyek ini bertujuan untuk merekonstruksi gambar bangunan yang realistis dari peta segmentasi (semantic segmentation) menggunakan model autoencoder berbasis **U-Net**. Model ini dilatih untuk menerjemahkan peta tata letak bangunan ke dalam gambar yang menyerupai bangunan nyata.

## 📁 Deskripsi Dataset
Setiap sampel data terdiri dari:

- **Input Image**: Gambar segmentasi di mana setiap warna mewakili bagian bangunan tertentu (misalnya: dinding, jendela, pintu).
- **Ground Truth**: Gambar nyata dari bangunan yang sesuai dengan segmentasi.
- **Predicted Image**: Gambar hasil prediksi model berdasarkan input segmentasi.

📉 **Perkembangan Training**:

![Cuplikan layar 2025-04-23 222451](https://github.com/user-attachments/assets/c2965b0f-4929-4f16-b69b-1087b3856fd8)

Hasil menunjukkan bahwa nilai loss semakin menurun seiring waktu, yang mengindikasikan bahwa model berhasil mempelajari hubungan antara input dan gambar aslinya.

## 📊 Hasil dan Evaluasi
![Cuplikan layar 2025-04-23 222318](https://github.com/user-attachments/assets/e15e2857-4019-429b-88c4-8f2208fcbf80)
![Cuplikan layar 2025-04-23 222312](https://github.com/user-attachments/assets/b309d76a-9ba7-4ace-8261-fe6f3ee6c1dc)
![Cuplikan layar 2025-04-23 222304](https://github.com/user-attachments/assets/e5817be8-e098-4d55-9aaf-c792d33952df)
![Cuplikan layar 2025-04-23 222257](https://github.com/user-attachments/assets/2eefd12d-5e84-4678-a7d3-255b26f3bfa4)
![Cuplikan layar 2025-04-23 222244](https://github.com/user-attachments/assets/eb00ab25-d1e1-4dbe-8820-5c5f6486c530)

Hasil prediksi menunjukkan bahwa model mampu merekonstruksi struktur bangunan yang mirip dengan gambar asli. Meskipun terdapat sedikit distorsi pada bagian tepi gambar, model dapat mengenali pola struktur dan tekstur secara umum.

**Peningkatan yang dapat dilakukan:**
- Menggunakan perceptual loss atau GAN (Generative Adversarial Network) untuk meningkatkan kualitas visual.
- Menambahkan data augmentasi untuk generalisasi yang lebih baik.
- Mencoba backbone encoder yang lebih dalam (seperti ResNet atau VGG).
