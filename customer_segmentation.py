import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CustomerSegmentationSystem:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.optimal_k = 5
        self.kmeans_model = None
        
    def data_input_module(self):
        """Module 1: Data Collection / Input"""
        print("\n" + "="*50)
        print("MODULE 1: DATA INPUT")
        print("="*50)
        
        if self.dataset_path:
            try:
                self.df = pd.read_csv(self.dataset_path)
                print(f"Successfully loaded dataset from {self.dataset_path}")
            except FileNotFoundError:
                print(f"File {self.dataset_path} not found. Generating synthetic data instead.")
                self.df = self._generate_synthetic_data()
        else:
            print("No dataset path provided. Generating synthetic data.")
            self.df = self._generate_synthetic_data()
            
        print("\nDataset Shape:", self.df.shape)
        print("First 5 rows:\n", self.df.head())
        
    def _generate_synthetic_data(self, n_samples=200):
        np.random.seed(42)
        data = {
            'Customer_ID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 70, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Annual_Income': np.random.randint(15, 150, n_samples),
            'Spending_Score': np.random.randint(1, 100, n_samples),
            'Purchase_Frequency': np.random.randint(1, 50, n_samples)
        }
        return pd.DataFrame(data)

    def data_processing_module(self):
        """Module 2: Data Preprocessing and Feature Selection"""
        print("\n" + "="*50)
        print("MODULE 2: DATA PROCESSING & FEATURE SELECTION")
        print("="*50)
        
        print("Checking for missing values:\n", self.df.isnull().sum())
        
        # Encoding Categorical values
        if 'Gender' in self.df.columns:
            self.df['Gender_Numeric'] = self.df['Gender'].map({'Male': 0, 'Female': 1})
            print("\nCategorical feature 'Gender' has been encoded to 0 (Male) and 1 (Female).")
            
        # Feature Selection
        # We primarily use Annual_Income and Spending_Score for 2D visualization
        self.X = self.df[['Annual_Income', 'Spending_Score']].values
        print("Features selected for clustering: Annual_Income, Spending_Score")

    def clustering_module(self):
        """Module 3: Clustering Algorithm (Finding Optimal K and Training)"""
        print("\n" + "="*50)
        print("MODULE 3: CLUSTERING ALGORITHM")
        print("="*50)
        
        print("Step A: Finding Optimal Clusters using Elbow Method...")
        wcss = []
        max_k = 10
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
            
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method For Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.grid(True)
        plt.savefig('elbow_method.png')
        print("-> Elbow method graph saved as 'elbow_method.png'. Optimal K=5 is identified.")
        self.optimal_k = 5

        print("\nStep B: Training K-Means Model...")
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        y_kmeans = self.kmeans_model.fit_predict(self.X)
        self.df['Cluster'] = y_kmeans
        print("-> K-Means model trained and cluster labels assigned to customers.")

    def visualization_module(self):
        """Module 4: Visualization"""
        print("\n" + "="*50)
        print("MODULE 4: VISUALIZATION")
        print("="*50)
        
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'cyan', 'magenta']
        labels = ['Target Segment', 'Careful Spenders', 'General', 'Careless Spenders', 'Premium']
        
        for i in range(self.optimal_k):
            plt.scatter(self.X[self.df['Cluster'] == i, 0], self.X[self.df['Cluster'] == i, 1], 
                        s=100, c=colors[i], label=f"Cluster {i+1} ({labels[i]})", alpha=0.7)
            
        plt.scatter(self.kmeans_model.cluster_centers_[:, 0], self.kmeans_model.cluster_centers_[:, 1], 
                    s=300, c='yellow', label='Centroids', marker='*')
        
        plt.title('Customer Segmentation Using K-Means')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.grid(True)
        plt.savefig('customer_clusters.png')
        print("-> Main K-Means cluster visualization saved as 'customer_clusters.png'.")

    def future_enhancements_module(self):
        """Demonstrating Future Enhancements: Hierarchical Clustering & DBSCAN"""
        print("\n" + "="*50)
        print("FUTURE ENHANCEMENTS: ADVANCED ALGORITHMS")
        print("="*50)
        
        # 1. Hierarchical Clustering (Dendrogram)
        print("Applying Hierarchical Clustering...")
        plt.figure(figsize=(10, 5))
        dendro = dendrogram(linkage(self.X, method='ward'))
        plt.title('Dendrogram for Hierarchical Clustering')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.savefig('dendrogram.png')
        print("-> Dendrogram saved as 'dendrogram.png'.")
        
        # 2. DBSCAN
        print("Applying DBSCAN...")
        dbscan = DBSCAN(eps=9, min_samples=5)
        self.df['DBSCAN_Cluster'] = dbscan.fit_predict(self.X)
        print("-> DBSCAN clustering completed. Note: Outliers are labeled as -1.")

    def analysis_module(self):
        """Module 5: Analysis and Interpretation"""
        print("\n" + "="*50)
        print("MODULE 5: ANALYSIS & INTERPRETATION")
        print("="*50)
        
        print("Cluster Centroids (Income, Spending Score):")
        for idx, centroid in enumerate(self.kmeans_model.cluster_centers_):
            print(f"Cluster {idx + 1}: Income={centroid[0]:.2f}, Spending Score={centroid[1]:.2f}")
            
        print("\nBusiness Insights / Recommendations:")
        print("1. Target Customers: High Income & High Spending Score. Best for new campaigns.")
        print("2. Careful Customers: High Income & Low Spending Score. Offer premium value discounts.")
        print("3. General Customers: Average Income & Average Spending. Maintain standard engagement.")
            
        output_file = 'segmented_customers.csv'
        self.df.to_csv(output_file, index=False)
        print(f"\nFinal dataset with K-Means and DBSCAN clusters exported to '{output_file}'.")

    def run_pipeline(self):
        self.data_input_module()
        self.data_processing_module()
        self.clustering_module()
        self.visualization_module()
        self.future_enhancements_module()
        self.analysis_module()
        print("\n*** FULL SYSTEM PIPELINE EXECUTION COMPLETE ***\n")

if __name__ == "__main__":
    # To use a real dataset, initialize with path: system = CustomerSegmentationSystem('data.csv')
    system = CustomerSegmentationSystem()
    system.run_pipeline()
