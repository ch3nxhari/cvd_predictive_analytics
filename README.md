# Laporan Proyek <em> Machine Learning </em> - Hari Pringadi

## Domain Proyek

Penyakit jantung adalah masalah kesehatan utama yang mempengaruhi jutaan orang di seluruh dunia. Ini adalah penyebab utama kematian di banyak negara, dan ada banyak faktor risiko yang terkait dengannya, termasuk tekanan darah tinggi, diabetes, merokok, obesitas, kebiasaan gaya hidup tidak sehat seperti kurangnya aktivitas fisik atau pilihan nutrisi yang buruk dan genetika. Penyakit kardiovaskular (CVD) juga memberikan beban finansial dan ekonomi yang signifikan bagi negara-negara anggota European Society of Cardiology (ESC). CVD diperkirakan menghabiskan €210 miliar per tahun bagi ekonomi Uni Eropa pada tahun 2015, dengan biaya perawatan kesehatan menyumbang 53%, kehilangan produktivitas 26%, dan perawatan informal 21%. Beban ekonomi akibat CVD bervariasi di beberapa negara, dengan biaya perawatan kesehatan langsung per kapita berkisar dari €48 di Bulgaria hingga €365 di Finlandia. Di Jerman, biaya CVD mencakup 13% dari total belanja kesehatan nasional. Data menunjukkan bahwa CVD bukan hanya masalah kesehatan, tetapi juga merupakan tantangan ekonomi bagi sistem kesehatan, dengan beban diperkirakan akan meningkat secara eksponensial di tahun-tahun mendatang.  Tujuan utama penulis adalah mengembangkan model prediktif yang dapat secara efektif memperkirakan prognosis berbagai penyakit berdasarkan fitur yang disediakan. Sehingga bisa memberikan pencegahan dini dengan mengidentifikasi potensi penyakit kardiovaskular yang akan berdampak pula pada penurunan biaya penanganan yang diakibatkan oleh penyakit kardiovaskular.[1]


## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara <em> preprocessing </em> pada data <em> Cardiovascular Diseases </em> yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi berbagai prognosis penyakit pada tingkat pasien?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan <em> preprocessing </em> data sehingga data tersebut siap untuk dilatih oleh model <em> Machine Learning </em>


- Untuk <em> preprocessing </em> data dapat dilakukan beberapa teknik, di antaranya :
    - Melakukan drop kolom pada kolom yang memiliki duplikat.
    - Melakukan pembagian dataset menjadi dua bagian dengan rasio 8:2 / 80% untuk train dan 20% untuk test.
    - Melakukan *Encoding*  dan *StandardScaler* dengan menggunakan *Pipeline*.
    
- Untuk Pemilihan model terbaik data dapat dilakukan beberapa teknik, diantaranya :
    - Menghitung metric yang akan menjadikan patokan kita untuk memilih model terbaik <em>(precision    , recall  ,  f1-score, support dan accuracy) </em>.
    
    - Berikut adalah Rumus untuk menghitung <em>precision</em>
    
        $$
        precision=TP/(TP+FP)
        $$
    
    - Berikut adalah rumus untuk menghitung *recall*
    
      $$
      recall=TP/(TP+FN)
      $$
    
    - Berikut adalah rumus untuk menghitung *f1-score*
    
        $$
        f1-score=2*(precision*recall/precision+recall)
        $$
    
    - Berikut adalah rumus untuk menghitung *accuracy*
        $$
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        $$
    
    - Rumus-rumus di atas dapat dihitung langsung menggunakan library python yaitu sklearn metrics

## Data Understanding
Data yang digunakan adalah data yang berasal dari Kaggle, data ini berisikan berbagai fitur yang terkait dengan kesehatan dan gaya hidup pasien. Berikut adalah datanya [<em>Cardiovascular Diseases Risk Prediction Dataset</em>](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset/code)

### Variabel-variabel pada *Cardiovascular Diseases Risk Prediction Dataset* adalah sebagai berikut:
- General_Health = Would you say that in general your health is—
- Checkup = About how long has it been since you last visited a doctor for a routine checkup?
- Exercise = During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?
- Heart_Disease = Respondents that reported having coronary heart disease or mycardialinfarction
- Skin_Cancer = Respondents that reported having skin cancer
- Other_Cancer = Respondents that reported having any other types of cancer
- Depression = Respondents that reported having a depressive disorder (including depression, major depression, dysthymia, or minor depression)
- Diabetes = Respondents that reported having a diabetes. If yes, what type of diabetes it is/was.
- Arthritis = Respondents that reported having an Arthritis
- Sex = Respondent's Gender
- Age_Category = Memiliki nilai range umur 18-80+.
- Height_(cm)_  = Memiliki nilai tinggi badan pasien.
- Weight_(kg)  = Memiliki nilai berat badan pasien.
- BMI = Memiliki nilai index massa tubuh pasien.
- Smoking_History = Memiliki nilai "Yes" dan "No".
- Alcohol_Consumption.
- Fruit_Consumption.
- Green_Vegetables_Consumption.
- FriedPotato_Consumption.



Dataset overview:

```
- Datasets Name :  Cardiovascular Diseases Risk Prediction Dataset
- Overall Columns:
    - Valid : 308854
    - MissMatched : 0
    - Missing : 0
- Source : The 2021 BRFSS Dataset from CDC
- Link : https://www.cdc.gov/brfss/annual_data/annual_2021.html
- License : CC0: Public Domain
```



### **EXPLORATORY DATA ANALYSIS**

### Analisis Univariat



![](assets\Gambar 1.png)

Gambar 1. Distribusi fitur numerikal

Hasil Analisis:

`Height_(cm)`: Tinggi pasien tampaknya mengikuti distribusi normal, dengan mayoritas pasien memiliki tinggi badan sekitar 160 hingga 180 cm.

`Weight_(kg)`: Berat pasien juga tampak terdistribusi normal, dengan sebagian besar pasien memiliki berat antara sekitar 60 dan 100 kg.

`BMI`: Distribusi Indeks Massa Tubuh agak condong ke kanan. Sejumlah besar pasien memiliki BMI antara 20 dan 30, yang termasuk dalam kisaran normal hingga kelebihan berat badan. Namun, ada juga sejumlah besar pasien dengan BMI dalam kisaran obesitas (>30).

`Alcohol_Consumption`: Fitur ini sangat condong ke kanan. Sebagian besar pasien memiliki konsumsi alkohol yang rendah, namun ada beberapa pasien yang konsumsinya tinggi.

`Fruit_Consumption`: Fitur ini juga condong ke kanan. Banyak pasien mengonsumsi buah-buahan secara teratur, tetapi banyak yang mengonsumsinya lebih jarang.

`Green_Vegetables_Consumption`: Fitur ini tampaknya terdistribusi secara normal, dengan sebagian besar pasien mengonsumsi sayuran hijau secukupnya.

`FriedPotato_Consumption`: Fitur ini condong ke kanan. Banyak pasien mengonsumsi kentang goreng lebih jarang, sementara beberapa mengonsumsinya lebih sering.



![](assets\Gambar 2.png)

Gambar 2. Distribusi fitur kategorikal

Hasil Analisis:

`General_Health`: Sebagian besar pasien menggambarkan kesehatan umum mereka sebagai "*Good*", dengan "*Very Good*" menjadi tanggapan paling umum kedua. Lebih sedikit pasien menilai kesehatan mereka sebagai "*Fair*" atau "*Poor*".

`Checkup`: Mayoritas pasien melakukan pemeriksaan dalam setahun terakhir. Lebih sedikit pasien yang melakukan pemeriksaan terakhir 2 tahun yang lalu atau lebih dari 5 tahun yang lalu.

`Exercise`: Lebih banyak pasien melaporkan bahwa mereka berolahraga dibandingkan dengan mereka yang tidak.

`Heart_Disease`: Sebagian besar pasien tidak memiliki penyakit jantung. Hanya sebagian kecil pasien yang memiliki penyakit jantung.

`Skin_Cancer`: Sebagian besar pasien tidak menderita kanker kulit.

`Other_Cancer`: Serupa dengan kanker kulit, sebagian besar pasien tidak memiliki bentuk kanker lain.

`Depression`: Sebagian besar pasien tidak menderita depresi. Namun, sejumlah pasien yang tidak sepele memang melaporkan mengalami depresi.

`Diabetes`: Serupa dengan ciri-ciri terkait penyakit di atas, kebanyakan pasien tidak menderita diabetes. Namun, sebagian kecil memang menderita diabetes.

`Arthritis`: Sebagian besar pasien tidak menderita artritis, tetapi banyak yang mengalaminya.

`Sex`: Jumlah pasien wanita sedikit lebih banyak daripada pasien pria dalam kumpulan data.

`Age_Category`: Kumpulan data mencakup pasien dari berbagai kategori usia. Kategori usia 50-54 memiliki pasien terbanyak, diikuti kategori 55-59 dan 60-64.

`Smoking_History`: Sebagian besar pasien tidak memiliki riwayat merokok.



### Analisis Bivariat



![](assets\Gambar 3.png)

Gambar 3. Hubungan antara kondisi penyakit dengan beberapa variabel pilihan

Hasil Analisis:

`Heart_Disease`:

Penyakit jantung lebih sering terjadi pada pasien yang menilai kesehatan umum mereka sebagai "*Poor*" atau "*Fair*".

Ini sedikit lebih umum pada pasien yang tidak berolahraga.

Pria lebih cenderung memiliki penyakit jantung daripada wanita.

Prevalensi penyakit jantung meningkat seiring bertambahnya usia, yang paling umum terjadi pada kategori usia 80+.

Penyakit jantung juga lebih sering terjadi pada pasien dengan riwayat merokok.

`Skin_Cancer`:

Kanker kulit lebih sering terjadi pada pasien yang menilai kesehatan umum mereka sebagai "*Good*" atau "*Very Good*".

Tidak banyak perbedaan prevalensi berdasarkan kebiasaan berolahraga.

Perempuan lebih mungkin untuk memiliki kanker kulit daripada laki-laki.

Prevalensi kanker kulit meningkat seiring bertambahnya usia, yang paling umum terjadi pada kategori usia 70-74 tahun.

Tidak banyak perbedaan prevalensi berdasarkan riwayat merokok.

`Other_Cancer`:

Kanker lain lebih sering terjadi pada pasien yang menilai kesehatan umum mereka sebagai "*Poor*" atau "*Fair*".

Mereka sedikit lebih umum pada pasien yang tidak berolahraga.

Tidak banyak perbedaan prevalensi berdasarkan jenis kelamin.

Prevalensi kanker lain meningkat seiring bertambahnya usia, dan paling sering terjadi pada kategori usia 75-79 tahun.

Kanker lain lebih sering terjadi pada pasien dengan riwayat merokok.

`Diabetes`:

Diabetes lebih umum pada pasien yang menilai kesehatan umum mereka sebagai "*Fair*" atau "*Poor*".

Hal ini lebih sering terjadi pada pasien yang tidak berolahraga.

Tidak banyak perbedaan prevalensi berdasarkan jenis kelamin.

Prevalensi diabetes meningkat seiring bertambahnya usia, yang paling umum terjadi pada kategori usia 70-74 tahun.

Diabetes lebih sering terjadi pada pasien dengan riwayat merokok.

`Arthritis`:

Arthritis lebih umum pada pasien yang menilai kesehatan umum mereka sebagai "*Fair*" atau "*Poor*".

Ini sedikit lebih umum pada pasien yang tidak berolahraga.

Wanita lebih cenderung menderita radang sendi daripada pria.

Prevalensi artritis meningkat seiring bertambahnya usia, paling sering terjadi pada kategori usia 75-79 tahun.

Arthritis sedikit lebih sering terjadi pada pasien dengan riwayat merokok.



### Analisis Multivariat



![](assets\Gambar 4.png)

Gambar 4. Hubungan antara kondisi penyakit, general health dan age category

Analisa Hasil:

Distribusi `General_Health` berdasarkan `Age_Category` menunjukkan bahwa dengan bertambahnya usia, proporsi individu yang menilai kesehatan mereka sebagai "*Good*" atau "*Very Good*" menurun, sedangkan proporsi yang menilai kesehatan mereka sebagai "*Fair*" atau "*Poor*" meningkat.

Hubungan antara `General_Health` dengan kondisi penyakit (`Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, `Diabetes`, `Arthritis`) menunjukkan beberapa pola yang menarik:

Untuk `Heart_Disease`, `Other_Cancer`, `Diabetes`, dan `Arthritis`, prevalensinya lebih tinggi di antara mereka yang menilai kesehatannya sebagai "*Poor*" atau "*Fair*". Hal ini menunjukkan bahwa kondisi ini secara signifikan dapat mempengaruhi persepsi individu kesehatan umum mereka.

Untuk `Skin_Cancer`, prevalensi tampaknya lebih merata di berbagai peringkat kesehatan. Ini dapat menunjukkan bahwa kanker kulit mungkin tidak mempengaruhi persepsi individu tentang kesehatan umum mereka sebanyak kondisi lainnya.



![](assets\Gambar 5.png)

Gambar 5. Hubungan antara kondisi penyakit, BMI dan exercise

Interpretasi Hasil:

Distribusi `BMI_Category` berdasarkan `Exercise` menunjukkan bahwa individu yang berolahraga memiliki proporsi BMI "Normal" yang lebih tinggi, sedangkan mereka yang tidak berolahraga memiliki proporsi BMI "*Overweight*" dan "*Obese*" yang lebih tinggi. Ini menunjukkan bahwa olahraga dikaitkan dengan tingkat BMI yang lebih sehat.

Hubungan antara `BMI_Category` dengan kondisi penyakit (`Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, `Diabetes`, `Arthritis`) menunjukkan pola sebagai berikut:

Untuk  `Heart_Disease`, `Diabetes`, dan `Arthritis`, prevalensinya lebih tinggi di antara mereka yang memiliki BMI "*Overweight*" dan "*Obese*". Ini menunjukkan bahwa kondisi ini mungkin terkait dengan tingkat BMI yang lebih tinggi.

Untuk `Skin_Cancer` dan `Other_Cancer`, prevalensi tampaknya lebih merata di berbagai `BMI_Category`. Ini bisa menunjukkan bahwa jenis kanker ini mungkin tidak terkait kuat dengan BMI seperti kondisi lainnya.

### Pembahasan dari hasil EDA

**Analisis Univariat:**
Distribusi variabel numerik, seperti `Height_(cm)`, `Weight_(kg)`, dan `BMI`, sebagian besar normal, dengan beberapa fitur seperti `Alcohol_Consumption`, `Fruit_Consumption`, dan `FriedPotato_Consumption` menunjukkan distribusi miring ke kanan. Hal ini menunjukkan bahwa sebagian besar pasien memiliki tingkat konsumsi yang rendah hingga sedang. 

Variabel kategori menunjukkan distribusi yang beragam. Misalnya, sebagian besar pasien menilai kesehatan umum mereka sebagai "Good" atau "Very Good" dan melakukan pemeriksaan terakhir dalam setahun terakhir. Selain itu, sebagian besar pasien melaporkan berolahraga secara teratur, dan sebagian besar tidak memiliki riwayat merokok. 

**Analisis Bivariat:** 
Analisis bivariat mengungkapkan hubungan antara fitur yang dipilih dan kondisi penyakit. Penyakit seperti  `Heart_Disease`,  `Other_Cancer`,  `Diabetes`, dan `Arthritis` lebih banyak terjadi pada pasien yang menilai kesehatan umum mereka "Poor" atau "Fair", tidak berolahraga, dan memiliki riwayat merokok. `Skin_Cancer`menunjukkan pola yang berbeda, lebih lazim pada pasien dengan kesehatan umum "Good" atau "Very Good" dan tidak menunjukkan perbedaan yang signifikan berdasarkan kebiasaan berolahraga. 

**Analisis Multivariat:** 
Analisis multivariat menunjukkan interaksi antara beberapa variabel. Misalnya, dengan bertambahnya usia, proporsi individu yang menilai kesehatan mereka sebagai "Good" atau "Very Good" menurun, sementara proporsi yang menilai kesehatan mereka sebagai "Fair" atau "Poor" meningkat. Demikian pula, individu yang berolahraga memiliki proporsi `BMI` "Normal" yang lebih tinggi, sementara mereka yang tidak berolahraga memiliki proporsi `BMI`"Overweight" dan "Obese" yang lebih tinggi.

**Analisis Korelasi:** 
Analisis korelasi mengungkapkan kekuatan dan arah hubungan antara fitur dan kondisi penyakit. `Age_Category` menunjukkan korelasi positif yang kuat dengan semua penyakit, menunjukkan bahwa risiko penyakit ini meningkat seiring bertambahnya usia. `Excercies` menunjukkan korelasi negatif, menunjukkan bahwa olahraga teratur dapat membantu mengurangi risiko penyakit ini. 

### Data Preparation

#### Correlation Matrix

![](assets\Gambar 6.png)

Gambar 6. Heatmap Korelasi

Hasil Analisis:

Heatmaps korelasi menyediakan representasi visual dari korelasi antara berbagai fitur dalam kumpulan data. Setiap kotak menunjukkan korelasi antara variabel pada setiap sumbu. Nilai korelasi berkisar dari -1 hingga 1. Nilai yang mendekati 1 menunjukkan korelasi positif yang kuat, nilai yang mendekati -1 menunjukkan korelasi negatif yang kuat, dan nilai di sekitar 0 menunjukkan tidak ada korelasi.

Berikut adalah beberapa pengamatan dari heatmaps:

`BMI`, `Weight_(kg)`, dan `Exercise` memiliki korelasi positif dengan `Diabetes`. Hal ini menunjukkan bahwa individu dengan BMI dan berat badan lebih tinggi atau yang tidak berolahraga lebih mungkin menderita diabetes.

`General_Health` memiliki korelasi negatif dengan `Diabetes`, `Heart_Disease`, `Arthritis`, dan `Depression`. Ini menunjukkan bahwa individu yang menilai kesehatan umum mereka buruk lebih cenderung memiliki kondisi ini.

`Age_Category` memiliki korelasi positif dengan  `Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, `Diabetes`, dan `Arthritis`. Ini menunjukkan bahwa risiko penyakit ini meningkat seiring bertambahnya usia.

`Sex_Male` memiliki korelasi positif dengan `Heart_Disease` dan korelasi negatif dengan `Arthritis` dan `Skin_Cancer`. Hal ini menunjukkan bahwa laki-laki lebih cenderung memiliki penyakit jantung tetapi lebih kecil kemungkinannya untuk menderita radang sendi atau kanker kulit.



![](assets\Gambar 7.png)

Gambar 7. Korelasi setiap fitur dengan variabel penyakit

Hasil Analisis:

Heatmaps korelasi menunjukkan korelasi setiap fitur dengan lima variabel penyakit:  `Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, `Diabetes`, dan `Arthritis`.

Dari heatmaps, kita dapat mengamati yang berikut:

`Heart_Disease`: Kondisi ini menunjukkan korelasi positif yang kuat dengan `Age_Category` dan `General_Health`, dan korelasi negatif dengan `Exercise` dan `Sex_Female`.

`Skin_Cancer`: Kondisi ini berkorelasi positif kuat dengan `Age_Category` dan `Sex_Male`, dan berkorelasi negatif dengan `Sex_Female`.

`Other_Cancer`: Kondisi ini menunjukkan korelasi positif yang kuat dengan `Age_Category` dan `General_Health`, dan korelasi negatif dengan `Sex_Female`.

`Diabetes`: Kondisi ini menunjukkan korelasi positif yang kuat dengan `Age_Category`, `General_Health`, dan `BMI`, dan korelasi negatif dengan `Exercise`.

Tabel 1. <em> Generative Describe Statistics </em>

| Parameters | **Height_(cm)** | Weight_(kg) | **BMI**   | **Alcohol_Consumption** | **Fruit_Consumption** | **Green_Vegetables_Consumption** | **FriedPotato_Consumption** |
| ---------- | --------------- | ----------- | --------- | ----------------------- | --------------------- | -------------------------------- | --------------------------- |
| count      | 308854.00       | 308854.00   | 308854.00 | 308854.00               | 308854.00             | 308854.00                        | 308854.00                   |
| mean       | 170.62          | 83.59       | 28.63     | 5.10                    | 29.84                 | 15.11                            | 6.30                        |
| std        | 10.66           | 21.34       | 6.52      | 8.20                    | 24.88                 | 14.93                            | 8.58                        |
| min        | 91.00           | 24.95       | 12.02     | 0.00                    | 0.00                  | 0.00                             | 0.00                        |
| 25%        | 163.00          | 68.04       | 24.21     | 0.00                    | 12.00                 | 4.00                             | 2.00                        |
| 50%        | 170.00          | 81.65       | 27.44     | 1.00                    | 30.00                 | 12.00                            | 4.00                        |
| 75%        | 178.00          | 95.25       | 31.85     | 6.00                    | 30.00                 | 20.00                            | 8.00                        |
| max        | 241.00          | 293.02      | 99.33     | 30.00                   | 120.00                | 128.00                           | 128.00                      |



![](assets\Gambar 8.png)Gambar 8. Boxplot kolom numerikal

- Hasil Analisis

    Ringkasan Statistik dan box plot menunjukkan bahwa ada beberapa outlier potensial dalam data numerik. Berikut adalah beberapa pengamatan:

    - `Height_(cm)`: Nilai minimum adalah 91 cm, dan maksimum adalah 241 cm. Ini bisa menjadi kasus yang ekstrem, tetapi perlu diselidiki lebih lanjut.
    - `Weight_(kg)`: Berat maksimum adalah 293,02 kg, yang tampaknya cukup tinggi. Ini berpotensi menjadi nilai outlier atau ekstrim.
    - `BMI`: BMI maksimum adalah 99,33, yang sangat tinggi, bahkan untuk kasus obesitas yang ekstrem. Ini mungkin menunjukkan kesalahan entri data.
    - `Alcohol_Consumption`: Nilai maksimumnya adalah 30, yang tampaknya cukup tinggi. Kita perlu memahami satuan pengukuran untuk menginterpretasikan apakah ini outlier atau tidak.
    - `Fruit_Consumption`, `Green_Vegetables_Consumption`, `FriedPotato_Consumption`: Nilai maksimum tampaknya cukup tinggi, tetapi tergantung pada unit pengukuran (misalnya, porsi per minggu/bulan).

      Outlier potensial dan nilai ekstrem ini harus diselidiki lebih lanjut untuk menentukan validitasnya dan kemungkinan dampaknya terhadap analisis.

#### Pipeline

- Sebelum melakukan pipeline lakukan splitting variable terlebih dahulu.

- Melakukan encoding pada fitur sebagai berikut:

  - Encode menggunakan OneHotEncoder pada kategorikal pipeline

  - Encode menggunakan FuctionTransformer dan StandardScaler pada numerikal pipeline

  - Encode menggunakan OrdinalEncoder pada Age category, General health dan Checkup.

  - Membuat daftar pipeline

  - Finalisasi *preprocessing pipeline*

    Berikut adalah final *preprocessing pipeline* yang akan digunakan pada model:


![](assets\Gambar 9.png)

​             Gambar 9. preprocessing pipeline

Diatas adalah rangkaian dalam pembuatan pipeline.[2]

## Modeling
#### Models

- Algoritma Penelitian ini melakukan pemodelan dengan 6 algoritma, yaitu AdaBoost, Decision Tree, K-Nearest Neighbour, Logistic Regression, LGBM, dan Random Forest.
    - Adaboost = AdaBoost (Adaptive Boosting) adalah teknik dalam machine learning dengan metode ensemble. Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (*decision trees*) satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 split. Pohon-pohon ini juga disebut *Decision Stumps*. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) secara berurutan sehingga membentuk suatu model yang kuat (strong ensemble learner). Parameter yang digunakan pada proyek ini adalah :
      - `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
      - `learning_rate` = *Learning rate* memperkuat kontribusi setiap regressor.
      - `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.[3]
    - Decision Tree = Decision Tree adalah jenis *supervised machine learning* yang digunakan untuk mengkategorikan atau membuat prediksi berdasarkan bagaimana sekumpulan pertanyaan sebelumnya dijawab. Model merupakan salah satu bentuk pembelajaran terawasi, artinya model dilatih dan diuji pada sekumpulan data yang berisi kategorisasi yang diinginkan. Parameter yang digunakan pada proyek ini adalah :
      - `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.[3]
    - K-Nearest =Neighbour K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Parameter yang digunakan pada proyek ini adalah :
      - `n_neighbors` = Jumlah k tetangga tedekat.
    - Logistic Regression = Logistic Regression adalah *supervised machine learning* yang digunakan untuk memprediksi variabel target kategorikal dependen. Parameter yang digunakan pada proyek ini adalah :
      - `max_iter` = menentukan jumlah iterasi maksimum yang akan dilakukan algoritma sebelum dianggap selesai.
      - `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.[3]
    - *LGBM* = LightGBM mengimplementasikan algoritma Gradient Boosting Decision Tree (GBDT) konvensional dengan penambahan dua teknik baru: Pengambilan Sampel Satu Sisi Berbasis Gradien (GOSS) dan Bundling Fitur Eksklusif (EFB). Teknik-teknik ini dirancang untuk secara signifikan meningkatkan efisiensi dan skalabilitas GBDT. Parameter yang digunakan pada proyek ini adalah :
      - `boosting_type` = mengatur jenis algoritma Boosting yang akan digunakan dalam model.
      - `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
      - `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
      - `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.[3]
    - Random Forest = Random Forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Parameter yang digunakan pada proyek ini adalah :
      - `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
      - `max_depth` = Kedalaman maksimum setiap tree.
      - `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.[3]
    
    Tabel 2. Model parameter
    
    | Model              | Parameters                                                   |
    | ------------------ | ------------------------------------------------------------ |
    | AdaBoost           | n_estimators=50, learning_rate=1.0, random_state=42          |
    | DecisionTree       | random_state=22                                              |
    | KNN                | n_neighbors=3                                                |
    | LogisticRegression | max_iter=10000, random_state=22                              |
    | LGBM               | boosting_type='gbdt', n_estimators=100, learning_rate=0.1, random_state=42 |
    | RandomForest       | n_estimators=100, random_state=22                            |
## Evaluation

Model yang digunakan adalah model klasifikasi. Pertama akan dilakukan evaluasi dengan ROC AUC. ROC AUC adalah semacam alat ukur performance untuk classification problem dalam menentukan threshold dari suatu model. [4]
### Receiver Operating Characteristic

![](assets\Gambar 10.png)

Gambar 10. ROC AUC pada setiap model

AUC (Area Under The ROC Curve): Nilai AUC pada *train* untuk setiap model memiliki nilai >= 0,70 - Ini berarti bahwa model memiliki peluang 70% untuk mengklasifikasikan contoh positif yang dipilih secara acak dengan benar sebagai lebih mungkin positif daripada contoh negatif yang dipilih secara acak.

AUC 1,0 mewakili model sempurna yang tidak membuat kesalahan, sedangkan AUC 0,5 mewakili model yang kinerjanya tidak lebih baik dari peluang acak. Oleh karena itu, AUC sebesar 0,70 adalah nilai yang cukup baik.


### Classification Report

Untuk melihat metric yang telah disebutkan maka ditampilkan dalam bentuk *classification report* sebagai berikut:

Tabel 3. <em> Classification Report </em>

|               Classification Report untuk AdaBoost: |     precision |     recall |     f1-score |     support |
| --------------------------------------------------: | ------------: | ---------: | -----------: | ----------: |
|                                                     |               |            |              |             |
|                                                   0 |          0.96 |       0.84 |         0.90 |       56761 |
|                                                   1 |          0.25 |       0.59 |         0.35 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.82 |       61755 |
|                                           macro avg |          0.60 |       0.72 |         0.62 |       61755 |
|                                        weighted avg |          0.90 |       0.82 |         0.85 |       61755 |
|                                                     |               |            |              |             |
|       **Classification Report untuk DecisionTree:** | **precision** | **recall** | **f1-score** | **support** |
|                                                     |               |            |              |             |
|                                                   0 |          0.93 |       0.91 |         0.92 |       56761 |
|                                                   1 |          0.20 |       0.26 |         0.23 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.86 |       61755 |
|                                           macro avg |          0.57 |       0.59 |         0.57 |       61755 |
|                                        weighted avg |          0.87 |       0.86 |         0.87 |       61755 |
|                                                     |               |            |              |             |
|                **Classification Report untuk KNN:** | **precision** | **recall** | **f1-score** | **support** |
|                                                     |               |            |              |             |
|                                                   0 |          0.95 |       0.82 |         0.88 |       56761 |
|                                                   1 |          0.19 |       0.48 |         0.28 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.79 |       61755 |
|                                           macro avg |          0.57 |       0.65 |         0.58 |       61755 |
|                                        weighted avg |          0.89 |       0.79 |         0.83 |       61755 |
|                                                     |               |            |              |             |
| **Classification Report untuk LogisticRegression:** | **precision** | **recall** | **f1-score** | **support** |
|                                                     |               |            |              |             |
|                                                   0 |          0.98 |       0.73 |         0.84 |       56761 |
|                                                   1 |          0.21 |       0.79 |         0.33 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.74 |       61755 |
|                                           macro avg |          0.59 |       0.76 |         0.58 |       61755 |
|                                        weighted avg |          0.91 |       0.74 |         0.80 |       61755 |
|                                                     |               |            |              |             |
|               **Classification Report untuk LGBM:** | **precision** | **recall** | **f1-score** | **support** |
|                                                     |               |            |              |             |
|                                                   0 |          0.92 |       0.99 |         0.96 |       56761 |
|                                                   1 |          0.41 |       0.06 |         0.10 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.92 |       61755 |
|                                           macro avg |          0.66 |       0.53 |         0.53 |       61755 |
|                                        weighted avg |          0.88 |       0.92 |         0.89 |       61755 |
|                                                     |               |            |              |             |
|       **Classification Report untuk RandomForest:** | **precision** | **recall** | **f1-score** | **support** |
|                                                     |               |            |              |             |
|                                                   0 |          0.93 |       0.98 |         0.95 |       56761 |
|                                                   1 |          0.37 |       0.11 |         0.17 |        4994 |
|                                                     |               |            |              |             |
|                                            accuracy |               |            |         0.91 |       61755 |
|                                           macro avg |          0.65 |       0.55 |         0.56 |       61755 |
|                                        weighted avg |          0.88 |       0.91 |         0.89 |       61755 |
|                                                     |               |            |              |             |

------

Dari classification report di atas bisa dilihat beberapa model memiliki nilai accuracy yang baik tetapi hanya LGBM yang memiliki nilai accuracy tertinggi.

### Conclusion

Dari hasil evaluasi, kita dapat mengetahui bahwa:

- DecisionTree memiliki F1-score tertinggi untuk kelas 1, tetapi recall-nya rendah, mungkin cenderung memprediksi kelas mayoritas (kelas 0).
- KNN dan RandomForest memiliki kombinasi nilai terbaik, menunjukkan performa yang baik dalam membedakan kelas.
- LogisticRegression memiliki recall tertinggi tetapi presisi yang rendah.

Jadi, kita mendapatkan dua model yang bisa dipilih antara KNN dan RandomForest. Bisa disimpulkan KNN bisa menjadi pilihan terbaik dengan dengan F1-score lebih tinggi dibanding RandomForest. 

## Referensi
[1]  A. Timmis *et al.*, “European society of cardiology: Cardiovascular disease statistics 2019,” *Eur Heart J*, vol. 41, no. 1, pp. 12–85, Jan. 2020, doi: 10.1093/eurheartj/ehz859. [accessed Jul. 26 2023]

[2]Santos, G. (2022, September 30). *A basic introduction to pipelines in Scikit learn*. Medium. https://towardsdatascience.com/a-basic-introduction-to-pipelines-in-scikit-learn-bd4cee34ad95   [accessed Jul. 26 2023]

[3] *Learn*. scikit. (n.d.-a). https://scikit-learn.org/stable/ [accessed Jul. 26 2023]

[4] Datasans. (2023, May 31). *Memahami Roc Dan Auc*. Medium. https://datasans.medium.com/memahami-roc-dan-auc-2e0e4f3638bf  [accessed Jul. 26 2023]
