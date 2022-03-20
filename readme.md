### ini merupakan pengujian untuk aplikasi bacaan_tulusan_arab menggunakan python jupyter notebook
### audio kemudiannya diload oleh librosa dengan menggunakan  librosa.load() dan menggunakan mfcc() untuk menconvert audio ke dalam bentuk mfcc 2D list
### dimana dengan menggunakan librosa akan dapat menampilkan visualisasi antara 2 audio dengan menggunakan librosa.display.specshow()
### audio yang diconvert menjadi mfcc 2D list akan diconvert menjadi 1D list untuk dicheck similarity dengan menggunakan consine similarity
### terdapat juga normalize distance yang mana jika mendekati 0 maka suara sama

### sound similarity metode menggunakan cosine similarity 