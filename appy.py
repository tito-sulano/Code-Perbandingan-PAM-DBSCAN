import streamlit as st  # Mengimpor library Streamlit untuk membuat aplikasi web interaktif
import pandas as pd  # Mengimpor library Pandas untuk manipulasi data
from sklearn_extra.cluster import KMedoids  # Mengimpor KMedoids dari sklearn_extra untuk clustering
from sklearn.cluster import DBSCAN  # Mengimpor DBSCAN dari sklearn untuk clustering
from sklearn.preprocessing import StandardScaler  # Mengimpor StandardScaler dari sklearn untuk normalisasi data
from sklearn.metrics import silhouette_score  # Mengimpor silhouette_score dari sklearn untuk evaluasi clustering
import matplotlib.pyplot as plt  # Mengimpor matplotlib.pyplot untuk visualisasi
import seaborn as sns  # Mengimpor seaborn untuk visualisasi data yang lebih menarik
import numpy as np  # Mengimpor numpy untuk komputasi numerik

class DataViewer:
    def __init__(self, df):
        self.df = df  # Inisialisasi objek dengan data frame

    # Menampilkan Data
    def display_data_preview(self):
        st.write("Data Preview:")  # Menampilkan teks "Data Preview" di aplikasi Streamlit
        st.write(self.df)  # Menampilkan data frame

class KMedoidsClustering:
    # Import data
    def __init__(self, df, X_scaled, selected_columns, scaler):
        self.df = df  # Inisialisasi objek dengan data frame
        self.X_scaled = X_scaled  # Data yang sudah dinormalisasi
        self.selected_columns = selected_columns  # Kolom yang dipilih untuk clustering
        self.scaler = scaler  # Scaler yang digunakan untuk normalisasi
        self.kmedoids_labels = None  # Label hasil clustering dengan KMedoids, awalnya None

    # Melakukan Clustering
    def perform_clustering(self):
        st.write("## PAM Clustering")  # Menampilkan teks "PAM Clustering" di aplikasi Streamlit
        silhouette_coefficients = []  # List untuk menyimpan nilai silhouette coefficient
        num_clusters = []  # List untuk menyimpan jumlah cluster
        max_sil = 0  # Variabel untuk menyimpan nilai silhouette coefficient terbaik

        # Mencari nilai silhouette coefficient terbaik
        for k in range(2, 11):  # Mencoba berbagai jumlah cluster dari 2 hingga 10
            kmedoids = KMedoids(n_clusters=k, init="random", random_state=5)  # Inisialisasi KMedoids dengan jumlah cluster k
            kmedoids.fit(self.X_scaled)  # Melakukan clustering
            n_clusters_ = len(set(kmedoids.labels_)) - (1 if -1 in kmedoids.labels_ else 0)  # Menghitung jumlah cluster
            num_clusters.append(n_clusters_)  # Menyimpan jumlah cluster
            score = silhouette_score(self.X_scaled, kmedoids.labels_)  # Menghitung silhouette coefficient
            silhouette_coefficients.append(score)  # Menyimpan silhouette coefficient
            if (score > max_sil) and (n_clusters_ >= 2):  # Jika score lebih besar dari max_sil dan jumlah cluster >= 2
                max_sil = score  # Update max_sil
                best_n_clusters = n_clusters_  # Update best_n_clusters

        # Menampilkan tabel nilai silhouette coefficient 
        kmedoids_data = pd.DataFrame({'num_clusters': num_clusters, 'silhouette_coefficients': silhouette_coefficients})  # Membuat data frame dengan jumlah cluster dan silhouette coefficient
        kmedoids_data.sort_values(by='silhouette_coefficients', inplace=True, ascending=False)  # Mengurutkan data berdasarkan silhouette coefficient secara menurun

        st.write("PAM Silhouette Coefficient")  # Menampilkan teks "PAM Silhouette Coefficient" di aplikasi Streamlit
        kmedoids_data = kmedoids_data[kmedoids_data['num_clusters'] >= 2]  # Menyaring data dengan jumlah cluster >= 2
        st.write(kmedoids_data)  # Menampilkan data frame

        st.set_option('deprecation.showPyplotGlobalUse', False)  # Mengatur opsi deprecation warning untuk pyplot
        plt.style.use("fivethirtyeight")  # Mengatur gaya plot
        plt.plot(range(2, 11), silhouette_coefficients)  # Membuat plot silhouette coefficient
        plt.xticks(range(2, 11))  # Mengatur nilai sumbu x
        plt.xlabel("Number of Clusters")  # Label sumbu x
        plt.ylabel("Silhouette Coefficients")  # Label sumbu y
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

        sc_kmedoid = max(silhouette_coefficients)  # Mendapatkan nilai silhouette coefficient maksimum
        st.write("Maximum Silhouette Coefficient:", sc_kmedoid)  # Menampilkan nilai silhouette coefficient maksimum
        st.write('Best Number of Clusters:', best_n_clusters)  # Menampilkan jumlah cluster terbaik

        kmedoids = KMedoids(n_clusters=best_n_clusters, random_state=0).fit(self.X_scaled)  # Melakukan clustering dengan jumlah cluster terbaik
        self.kmedoids_labels = kmedoids.labels_  # Menyimpan label hasil clustering

        st.write("PAM Cluster Table")  # Menampilkan teks "PAM Cluster Table" di aplikasi Streamlit
        df_cluster_with_kmediods = pd.concat([self.df[self.selected_columns], pd.DataFrame({'cluster': self.kmedoids_labels})], axis=1)  # Menggabungkan data frame dengan label cluster
        st.write(df_cluster_with_kmediods)  # Menampilkan data frame dengan label cluster

        st.write("PAM Cluster Distribution:")  # Menampilkan teks "PAM Cluster Distribution" di aplikasi Streamlit
        st.write(pd.DataFrame({'cluster': self.kmedoids_labels}).value_counts())  # Menampilkan distribusi cluster

        sns.countplot(x=df_cluster_with_kmediods.cluster)  # Membuat countplot distribusi cluster
        st.set_option('deprecation.showPyplotGlobalUse', False)  # Mengatur opsi deprecation warning untuk pyplot
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

        self.kmedoids_labels = kmedoids.labels_  # Menyimpan label hasil clustering
        return self.kmedoids_labels  # Mengembalikan label hasil clustering

    def visualize_cluster(self):
        st.write("Visualisasi Cluster PAM")  # Menampilkan teks "Visualisasi Cluster PAM" di aplikasi Streamlit
        pam_cluster = pd.DataFrame({'cluster': self.kmedoids_labels})  # Membuat data frame dengan label cluster
        df_with_cluster = pd.concat([self.df[self.selected_columns], pam_cluster], axis=1)  # Menggabungkan data frame dengan label cluster
        sns.pairplot(df_with_cluster, hue='cluster')  # Membuat pairplot dengan label cluster
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

class DBSCANClustering:
    def __init__(self, df, X_scaled, selected_columns, scaler):
        self.df = df  # Inisialisasi objek dengan data frame
        self.X_scaled = X_scaled  # Data yang sudah dinormalisasi
        self.selected_columns = selected_columns  # Kolom yang dipilih untuk clustering
        self.scaler = scaler  # Scaler yang digunakan untuk normalisasi
        self.dbscan_labels = None  # Label hasil clustering dengan DBSCAN, awalnya None

    def perform_clustering(self):
        st.write("## DBSCAN Clustering")  # Menampilkan teks "DBSCAN Clustering" di aplikasi Streamlit
        Eps = []  # List untuk menyimpan nilai eps
        Min_samples = []  # List untuk menyimpan nilai min_samples
        num_clusters = []  # List untuk menyimpan jumlah cluster
        silhouette_coefficients = []  # List untuk menyimpan nilai silhouette coefficient
        noise = []  # List untuk menyimpan jumlah noise
        max_sil = 0  # Variabel untuk menyimpan nilai silhouette coefficient terbaik

        # Mencari nilai silhouette coefficient terbaik
        for i in range(2, 10):  # Mencoba berbagai nilai min_samples dari 2 hingga 9
            for j in np.arange(0.1, 1.0, 0.1):  # Mencoba berbagai nilai eps dari 0.1 hingga 0.9
                model_db = DBSCAN(eps=j, min_samples=i, metric='euclidean')  # Inisialisasi DBSCAN dengan nilai eps dan min_samples tertentu
                model_db.fit(self.X_scaled)  # Melakukan clustering
                if len(set(model_db.labels_)) == 1:  # Jika hanya ada satu cluster, lanjut ke iterasi berikutnya
                    continue
                labels = model_db.labels_  # Mendapatkan label hasil clustering
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # Menghitung jumlah cluster
                n_noise = len(labels[labels == -1])  # Menghitung jumlah noise
                noise.append(n_noise)  # Menyimpan jumlah noise
                num_clusters.append(n_clusters_)  # Menyimpan jumlah cluster
                Min_samples.append(i)  # Menyimpan nilai min_samples
                Eps.append(j)  # Menyimpan nilai eps

                score = silhouette_score(self.X_scaled, model_db.labels_)  # Menghitung silhouette coefficient
                silhouette_coefficients.append(score)  # Menyimpan silhouette coefficient
                if (score > max_sil) and (n_clusters_ >= 2):  # Jika score lebih besar dari max_sil dan jumlah cluster >= 2
                    max_sil = score  # Update max_sil
                    best_n_clusters = n_clusters_  # Update best_n_clusters
                    min_sample = i  # Update min_sample
                    ep = j  # Update ep

        dbscan_data = pd.DataFrame({'num_clusters': num_clusters, 'silhouette_coefficients': silhouette_coefficients,
                                    'eps': Eps, 'min_samples': Min_samples, 'noise': noise})  # Membuat data frame dengan nilai eps, min_samples, jumlah cluster, silhouette coefficient, dan jumlah noise
        dbscan_data.sort_values(by='silhouette_coefficients', inplace=True, ascending=False)  # Mengurutkan data berdasarkan silhouette coefficient secara menurun

        st.write("DBSCAN Silhouette Coefficient:")  # Menampilkan teks "DBSCAN Silhouette Coefficient" di aplikasi Streamlit
        dbscan_data = dbscan_data[dbscan_data['num_clusters'] >= 2]  # Menyaring data dengan jumlah cluster >= 2
        st.write(dbscan_data)  # Menampilkan data frame

        plt.subplots(figsize=(12, 6))  # Membuat subplots dengan ukuran tertentu
        plt.style.use("fivethirtyeight")  # Mengatur gaya plot
        plt.plot(dbscan_data['num_clusters'], dbscan_data['silhouette_coefficients'])  # Membuat plot silhouette coefficient
        plt.xticks(num_clusters)  # Mengatur nilai sumbu x
        plt.xlabel("Number of Clusters")  # Label sumbu x
        plt.ylabel("Silhouette Coefficients")  # Label sumbu y
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

        st.write("Maximum Silhouette Coefficient:", max_sil)  # Menampilkan nilai silhouette coefficient maksimum
        st.write('Minimal Samples:', min_sample)  # Menampilkan nilai min_samples terbaik
        st.write('Best Eps:', ep)  # Menampilkan nilai eps terbaik

        dbscan = DBSCAN(eps=ep, min_samples=min_sample, metric='euclidean').fit(self.X_scaled)  # Melakukan clustering dengan nilai eps dan min_samples terbaik
        dbscan_labels = dbscan.labels_  # Mendapatkan label hasil clustering

        st.write("DBSCAN Cluster Table")  # Menampilkan teks "DBSCAN Cluster Table" di aplikasi Streamlit
        df_cluster_with_dbscan = pd.concat([self.df[self.selected_columns], pd.DataFrame({'cluster': dbscan_labels})], axis=1)  # Menggabungkan data frame dengan label cluster
        st.write(df_cluster_with_dbscan)  # Menampilkan data frame dengan label cluster

        st.write("Distribusi Cluster DBSCAN:")  # Menampilkan teks "Distribusi Cluster DBSCAN" di aplikasi Streamlit
        st.write(pd.DataFrame({'cluster': dbscan_labels}).value_counts())  # Menampilkan distribusi cluster

        sns.countplot(x=df_cluster_with_dbscan.cluster)  # Membuat countplot distribusi cluster
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

        self.dbscan_labels = dbscan.labels_  # Menyimpan label hasil clustering
        return self.dbscan_labels  # Mengembalikan label hasil clustering

    def visualize_cluster(self):
        st.write("Visualisasi Cluster PAM")  # Menampilkan teks "Visualisasi Cluster PAM" di aplikasi Streamlit
        dbscan_cluster = pd.DataFrame({'cluster': self.dbscan_labels})  # Membuat data frame dengan label cluster
        df_with_cluster_dbscan = pd.concat([self.df[self.selected_columns], dbscan_cluster], axis=1)  # Menggabungkan data frame dengan label cluster
        sns.pairplot(df_with_cluster_dbscan, hue='cluster')  # Membuat pairplot dengan label cluster
        st.pyplot()  # Menampilkan plot di aplikasi Streamlit

class SilhouetteCoefficientCalculator:
    def __init__(self, X_scaled, kmedoids_labels, dbscan_labels):
        self.X_scaled = X_scaled  # Data yang sudah dinormalisasi
        self.kmedoids_labels = kmedoids_labels  # Label hasil clustering dengan KMedoids
        self.dbscan_labels = dbscan_labels  # Label hasil clustering dengan DBSCAN

    def calculate_silhouette_coefficient(self):
        if self.kmedoids_labels is not None:
            kmedoids_silhouette = silhouette_score(self.X_scaled, self.kmedoids_labels)  # Menghitung silhouette coefficient untuk KMedoids
        else:
            kmedoids_silhouette = None
        
        if self.dbscan_labels is not None:
            dbscan_silhouette = silhouette_score(self.X_scaled, self.dbscan_labels)  # Menghitung silhouette coefficient untuk DBSCAN
        else:
            dbscan_silhouette = None
        
        return kmedoids_silhouette, dbscan_silhouette  # Mengembalikan silhouette coefficient untuk KMedoids dan DBSCAN

def main():
    st.title("PAM & DBSCAN Algorithms Comparison")  # Menampilkan judul aplikasi Streamlit

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])  # Widget untuk mengunggah file CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # Membaca file CSV yang diunggah

        data_viewer = DataViewer(df)  # Membuat objek DataViewer
        data_viewer.display_data_preview()  # Menampilkan pratinjau data

        selected_columns = st.multiselect("Select columns", df.columns.tolist())  # Widget untuk memilih kolom

        if st.button("Cluster"):  # Tombol untuk melakukan clustering
            if len(selected_columns) > 1:
                X = df[selected_columns]  # Mengambil data berdasarkan kolom yang dipilih
                scaler = StandardScaler()  # Membuat objek StandardScaler
                X_scaled = scaler.fit_transform(X)  # Melakukan normalisasi data

                kmedoids_clustering = KMedoidsClustering(df, X_scaled, selected_columns, scaler)  # Membuat objek KMedoidsClustering
                kmedoids_labels = kmedoids_clustering.perform_clustering()  # Melakukan clustering dengan KMedoids
                kmedoids_clustering.visualize_cluster()  # Visualisasi hasil clustering dengan KMedoids

                dbscan_clustering = DBSCANClustering(df, X_scaled, selected_columns, scaler)  # Membuat objek DBSCANClustering
                dbscan_labels = dbscan_clustering.perform_clustering()  # Melakukan clustering dengan DBSCAN
                dbscan_clustering.visualize_cluster()  # Visualisasi hasil clustering dengan DBSCAN

                silhouette_coefficient_calculator = SilhouetteCoefficientCalculator(X_scaled, kmedoids_labels, dbscan_labels)  # Membuat objek SilhouetteCoefficientCalculator
                kmedoids_silhouette, dbscan_silhouette = silhouette_coefficient_calculator.calculate_silhouette_coefficient()  # Menghitung silhouette coefficient untuk KMedoids dan DBSCAN

                df_silhouette = pd.DataFrame({'Algorithm': ['PAM', 'DBSCAN'], 'Silhouette Coefficient': [kmedoids_silhouette, dbscan_silhouette]})  # Membuat data frame untuk membandingkan silhouette coefficient
                st.write("Silhouette Coefficient Comparison:")  # Menampilkan teks "Silhouette Coefficient Comparison" di aplikasi Streamlit
                st.write(df_silhouette)  # Menampilkan data frame dengan silhouette coefficient

                plt.figure(figsize=(8, 6))  # Membuat figure dengan ukuran tertentu
                sns.barplot(x='Algorithm', y='Silhouette Coefficient', data=df_silhouette)  # Membuat barplot untuk membandingkan silhouette coefficient
                plt.xlabel('Algorithm')  # Label sumbu x
                plt.ylabel('Silhouette Coefficient')  # Label sumbu y
                plt.title('Silhouette Coefficient Comparison')  # Judul plot
                st.pyplot()  # Menampilkan plot di aplikasi Streamlit

            else:
                st.error("Please select at least 2 columns for clustering")  # Menampilkan pesan error jika kolom yang dipilih kurang dari 2

if __name__ == "__main__":
    main()  # Menjalankan fungsi main jika script dijalankan langsung




    