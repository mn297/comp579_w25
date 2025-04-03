import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

use_gpu = False
if use_gpu:
    from cuml.cluster import KMeans  # cuML's KMeans
    from cuml.metrics import silhouette_score, davies_bouldin_score  # cuML's metrics
else:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
import cupy as cp
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm


# Group data by increments of 0.005
def plot_ticks(df):
    bins = pd.cut(
        df["close"],
        bins=pd.interval_range(
            start=df["close"].min(), end=df["close"].max(), freq=0.0005
        ),
    )
    tick_counts = df.groupby(bins)["close"].count()
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))

    # Get the mid-point of each interval for plotting
    mid_points = [interval.mid for interval in tick_counts.index.categories]

    # plt.scatter(mid_points, tick_counts.values, alpha=0.5, color='#6082B6', s=8)
    plt.bar(mid_points, tick_counts.values, width=0.0004, alpha=0.7, color="#4169E1")

    plt.xlabel("Closing Prices")
    plt.ylabel("Tick Counts")
    plt.title("EUR/USD Tick Counts per Closing Price")
    plt.xticks(rotation=45)  # Rotate x-labels for better visibility
    plt.show()


def find_optimal_clustering(df):
    max_clusters = 20
    silhouette_scores = []
    db_scores = []
    sse = []  # For Elbow Method

    data = df["close"].values.reshape(-1, 1)

    # Calculate metrics for different numbers of clusters with progress bar
    for k in tqdm(range(5, max_clusters + 1), desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_

        # Silhouette Score
        silhouette_scores.append(silhouette_score(data, labels))

        # Davies-Bouldin Score
        db_scores.append(davies_bouldin_score(data, labels))

        # SSE for Elbow Method
        sse.append(kmeans.inertia_)

    # Prepare x-axis values
    k_range = list(range(5, max_clusters + 1))

    # Create figure and axes using object-oriented interface
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Plot 1: Silhouette Scores (higher is better)
    axes[0].plot(k_range, silhouette_scores, marker="o", linestyle="-", color="blue")
    axes[0].set_title("Silhouette Scores", fontsize=14)
    axes[0].set_xlabel("Number of Clusters", fontsize=12)
    axes[0].set_ylabel("Silhouette Score", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Find and highlight the best silhouette score
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    axes[0].axvline(x=best_k_silhouette, color="r", linestyle="--", alpha=0.5)
    axes[0].text(
        best_k_silhouette + 0.5,
        max(silhouette_scores),
        f"Best k={best_k_silhouette}",
        color="red",
        fontsize=10,
    )

    # Plot 2: Davies-Bouldin Scores (lower is better)
    axes[2].plot(k_range, db_scores, marker="o", linestyle="-", color="green")
    axes[2].set_title("Davies-Bouldin Scores", fontsize=14)
    axes[2].set_xlabel("Number of Clusters", fontsize=12)
    axes[2].set_ylabel("Davies-Bouldin Score", fontsize=12)
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # Find and highlight the best Davies-Bouldin score
    best_k_db = k_range[np.argmin(db_scores)]
    axes[2].axvline(x=best_k_db, color="r", linestyle="--", alpha=0.5)
    axes[2].text(
        best_k_db + 0.5, min(db_scores), f"Best k={best_k_db}", color="red", fontsize=10
    )

    # Plot 3: Elbow Method (SSE)
    axes[3].plot(k_range, sse, marker="o", linestyle="-", color="purple")
    axes[3].set_title("Elbow Method", fontsize=14)
    axes[3].set_xlabel("Number of Clusters", fontsize=12)
    axes[3].set_ylabel("Sum of Squared Errors (SSE)", fontsize=12)
    axes[3].grid(True, linestyle="--", alpha=0.7)

    # Optional: calculate second derivative to find the elbow point automatically
    sse_diff = np.diff(sse, 2)
    if len(sse_diff) > 0:
        elbow_index = (
            np.argmax(sse_diff) + 5 + 1
        )  # +5 for the range start, +1 for the diff offset
        axes[3].axvline(x=elbow_index, color="r", linestyle="--", alpha=0.5)
        axes[3].text(
            elbow_index + 0.5,
            sse[elbow_index - 5],
            f"Elbow k={elbow_index}",
            color="red",
            fontsize=10,
        )

    # Remove the empty subplot
    fig.delaxes(axes[1])

    # Add a title for the entire figure
    fig.suptitle("Optimal Number of Clusters Analysis", fontsize=16, y=0.98)

    # Adjust layout and spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Return the best k values according to different metrics
    best_k = {
        "silhouette": best_k_silhouette,
        "davies_bouldin": best_k_db,
        "elbow": elbow_index if len(sse_diff) > 0 else None,
    }

    plt.show()
    return best_k


if __name__ == "__main__":
    df = pd.read_csv("FX_EURUSD30.csv")
    best_k = find_optimal_clustering(df)
    print(f"Best k values: {best_k}")


def kmeans(data_30m, data_5m, clusters):
    def calculate_reversals(prices, centers, look_forward=4, reversal_threshold=0.002):
        reversals = {center: 0 for center in centers}
        for i in range(len(prices)):
            for center in centers:
                if abs(prices[i] - center) / center <= margin_of_error:
                    for j in range(i + 1, min(i + 1 + look_forward, len(prices))):
                        if abs(prices[j] - prices[i]) / prices[i] >= reversal_threshold:
                            reversals[center] += 1
                            break
        return reversals

    # Load data
    prices = data_30m["close"].values.reshape(-1, 1)
    train_prices, test_prices_30m = train_test_split(
        prices, test_size=0.075, shuffle=False
    )
    test_prices = data_5m[data_5m["time"] >= 1696867200]["close"].values

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(
        data_30m[data_30m["time"] < 1696867200][["close", "open", "high", "low"]].values
    )

    # Extract the cluster centers
    centers = kmeans.cluster_centers_[
        :, 0
    ]  # Assuming you want to use the 'close' price as the center

    # Define margin of error
    margin_of_error = 0.003  # ±0.3%

    # Calculate hits on the test set
    hits = {center: 0 for center in centers}
    for price in test_prices:
        for center in centers:
            if abs(price - center) / center <= margin_of_error:
                hits[center] += 1

    num_simulations = 100
    total_hits_model, total_reversals_model = np.zeros(len(centers)), np.zeros(
        len(centers)
    )
    total_hits_wiener, total_reversals_wiener = np.zeros(len(centers)), np.zeros(
        len(centers)
    )

    for seed in range(num_simulations):
        np.random.seed(seed)

        # Re-simulate Wiener process
        wiener_process = np.empty_like(test_prices)
        wiener_process[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_process[i] = wiener_process[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )

        # Calculate hits and reversals for actual data
        hits = {center: 0 for center in centers}
        actual_reversals = calculate_reversals(test_prices, centers)
        for price in test_prices:
            for center in centers:
                if abs(price - center) / center <= margin_of_error:
                    hits[center] += 1

        # Calculate hits and reversals for Wiener process
        wiener_hits = {center: 0 for center in centers}
        wiener_reversals = calculate_reversals(wiener_process, centers)
        for price in wiener_process:
            for center in centers:
                if abs(price - center) / center <= margin_of_error:
                    wiener_hits[center] += 1

        # Accumulate hits and reversals
        for i, center in enumerate(centers):
            total_hits_model[i] += hits[center]
            total_reversals_model[i] += actual_reversals[center]
            total_hits_wiener[i] += wiener_hits[center]
            total_reversals_wiener[i] += wiener_reversals[center]

    # Calculate mean hit and reversal rates, excluding levels with zero hits or reversals
    mean_hits_model = total_hits_model / num_simulations
    mean_reversals_model = total_reversals_model / num_simulations
    mean_hits_wiener = total_hits_wiener / num_simulations
    mean_reversals_wiener = total_reversals_wiener / num_simulations

    # Exclude levels with zero total hits or total reversals from mean values
    non_zero_indices_hits = np.where(mean_hits_model != 0)[0]
    non_zero_indices_reversals = np.where(mean_reversals_model != 0)[0]

    # Extract non-zero centers
    non_zero_centers_hits = [centers[i] for i in non_zero_indices_hits]
    non_zero_centers_reversals = [centers[i] for i in non_zero_indices_reversals]

    # Plotting with mean hit and reversal rates (excluding zero values)
    plt.figure(figsize=(15, 8))

    # Plot train data, test data, and the first 10 Wiener processes
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_prices)), train_prices, label="Train Data")
    plt.plot(
        range(len(train_prices), len(train_prices) + len(test_prices)),
        test_prices,
        label="Test Data",
    )
    for seed in range(10):
        np.random.seed(seed)
        wiener_temp = np.empty_like(test_prices)
        wiener_temp[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_temp[i] = wiener_temp[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )
        plt.plot(
            range(len(train_prices), len(train_prices) + len(wiener_temp)),
            wiener_temp,
            alpha=0.15,
        )
    for level in centers:
        plt.axhline(y=level, color="red", linestyle="--")
    plt.title("Train Data, Test Data, and Wiener Processes")
    plt.legend()

    # Plot for mean hit counts comparison (excluding zero values)
    plt.subplot(2, 2, 2)
    width = 0.35
    indices_hits = np.arange(len(non_zero_centers_hits))
    plt.bar(
        indices_hits - width / 2,
        mean_hits_model[non_zero_indices_hits],
        width,
        label="Model Mean Hits",
    )
    plt.bar(
        indices_hits + width / 2,
        mean_hits_wiener[non_zero_indices_hits],
        width,
        label="Wiener Process Mean Hits",
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Hit Counts")
    plt.title("Mean Hit Counts Comparison")
    plt.xticks(
        indices_hits, [f"{center:.4f}" for center in non_zero_centers_hits], rotation=45
    )
    plt.legend()

    # Plot for mean reversal counts comparison (excluding zero values)
    plt.subplot(2, 1, 2)
    indices_reversals = np.arange(len(non_zero_centers_reversals))
    plt.bar(
        indices_reversals - width / 2,
        mean_reversals_model[non_zero_indices_reversals],
        width,
        label="Model Mean Reversals",
    )
    plt.bar(
        indices_reversals + width / 2,
        mean_reversals_wiener[non_zero_indices_reversals],
        width,
        label="Wiener Process Mean Reversals",
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Reversal Counts")
    plt.title("Mean Reversal Counts Comparison")
    plt.xticks(
        indices_reversals,
        [f"{center:.4f}" for center in non_zero_centers_reversals],
        rotation=45,
    )
    plt.legend()

    # Statistical comparison for hit rate (excluding zero values)
    non_zero_hits_model = np.array(list(hits.values()))[non_zero_indices_hits]
    non_zero_wiener_hits = np.array(list(wiener_hits.values()))[non_zero_indices_hits]
    t_stat, p_value = stats.ttest_ind(non_zero_hits_model, non_zero_wiener_hits)
    print(
        "Model Hit Rate:",
        sum(non_zero_hits_model) / (len(test_prices) * len(non_zero_centers_hits)),
    )
    print(
        "Wiener Process Hit Rate:",
        sum(non_zero_wiener_hits) / (len(wiener_process) * len(non_zero_centers_hits)),
    )
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    # Statistical comparison for reversal rate (excluding zero values)
    non_zero_actual_reversals = np.array(list(actual_reversals.values()))[
        non_zero_indices_reversals
    ]
    non_zero_wiener_reversals = np.array(list(wiener_reversals.values()))[
        non_zero_indices_reversals
    ]
    t_stat_reversals, p_value_reversals = stats.ttest_ind(
        non_zero_actual_reversals, non_zero_wiener_reversals
    )
    print(
        "\n\nModel Reversal Rate:",
        sum(non_zero_actual_reversals)
        / (len(test_prices) * len(non_zero_centers_reversals)),
    )
    print(
        "Wiener Process Reversal Rate:",
        sum(non_zero_wiener_reversals)
        / (len(wiener_process) * len(non_zero_centers_reversals)),
    )
    print("Reversal t-statistic:", t_stat_reversals)
    print("Reversal p-value:", p_value_reversals)

    plt.tight_layout()
    plt.show()


def gmm(data_30m, data_5m, clusters):
    def calculate_reversals(prices, centers, look_forward=4, reversal_threshold=0.002):
        reversals = {center[0]: 0 for center in centers}
        for i in range(len(prices)):
            for center in centers:
                if abs(prices[i] - center[0]) / center[0] <= margin_of_error:
                    for j in range(i + 1, min(i + 1 + look_forward, len(prices))):
                        if abs(prices[j] - prices[i]) / prices[i] >= reversal_threshold:
                            reversals[center[0]] += 1
                            break
        return reversals

    # Load 30-minute data and split
    prices_30m = data_30m["close"].values.reshape(-1, 1)
    train_prices, test_prices_30m = train_test_split(
        prices_30m, test_size=0.075, shuffle=False
    )
    test_prices = data_5m[data_5m["time"] >= 1696867200]["close"].values

    # Cluster with GMM on the training set
    gmm = GaussianMixture(n_components=clusters, random_state=10)
    gmm.fit(
        data_30m[data_30m["time"] < 1696867200][["close", "open", "high", "low"]].values
    )
    centers = gmm.means_

    # Define margin of error
    margin_of_error = 0.003  # ±0.3%

    # Calculate hits on the test set
    hits = {center[0]: 0 for center in centers}
    for price in test_prices:
        for center in centers:
            if abs(price - center[0]) / center[0] <= margin_of_error:
                hits[center[0]] += 1

    # Exclude cluster levels with zero hits in the test data
    non_zero_centers = [center for center in centers if hits[center[0]] > 0]
    non_zero_indices = [i for i, center in enumerate(centers) if hits[center[0]] > 0]

    # Simulation code
    num_simulations = 100
    total_hits_model, total_reversals_model = np.zeros(len(centers)), np.zeros(
        len(centers)
    )
    total_hits_wiener, total_reversals_wiener = np.zeros(len(centers)), np.zeros(
        len(centers)
    )

    for seed in range(num_simulations):
        np.random.seed(seed)

        # Re-simulate Wiener process
        wiener_process = np.empty_like(test_prices)
        wiener_process[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_process[i] = wiener_process[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )

        # Calculate hits and reversals for actual data
        hits = {center[0]: 0 for center in centers}
        actual_reversals = calculate_reversals(test_prices, centers)
        for price in test_prices:
            for center in centers:
                if abs(price - center[0]) / center[0] <= margin_of_error:
                    hits[center[0]] += 1

        # Calculate hits and reversals for Wiener process
        wiener_hits = {center[0]: 0 for center in centers}
        wiener_reversals = calculate_reversals(wiener_process, centers)
        for price in wiener_process:
            for center in centers:
                if abs(price - center[0]) / center[0] <= margin_of_error:
                    wiener_hits[center[0]] += 1

        # Accumulate hits and reversals
        for i, center in enumerate(centers):
            total_hits_model[i] += hits[center[0]]
            total_reversals_model[i] += actual_reversals[center[0]]
            total_hits_wiener[i] += wiener_hits[center[0]]
            total_reversals_wiener[i] += wiener_reversals[center[0]]

    # Calculate mean hit and reversal rates for non-zero centers
    mean_hits_model = (total_hits_model / num_simulations)[non_zero_indices]
    mean_reversals_model = (total_reversals_model / num_simulations)[non_zero_indices]
    mean_hits_wiener = (total_hits_wiener / num_simulations)[non_zero_indices]
    mean_reversals_wiener = (total_reversals_wiener / num_simulations)[non_zero_indices]

    # Plotting
    plt.figure(figsize=(15, 8))

    # Plot train data, test data, and Wiener processes
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_prices)), train_prices, label="Train Data")
    plt.plot(
        range(len(train_prices), len(train_prices) + len(test_prices)),
        test_prices,
        label="Test Data",
    )
    for seed in range(10):
        np.random.seed(seed)
        wiener_temp = np.empty_like(test_prices)
        wiener_temp[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_temp[i] = wiener_temp[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )
        plt.plot(
            range(len(train_prices), len(train_prices) + len(wiener_temp)),
            wiener_temp,
            alpha=0.15,
        )
    for level in centers:
        plt.axhline(y=level[0], color="red", linestyle="--")
    plt.title("Train Data, Test Data, and Wiener Processes")
    plt.legend()

    # Plot for mean hit counts comparison (excluding zero values)
    plt.subplot(2, 2, 2)
    width = 0.35
    indices = np.arange(len(non_zero_centers))
    plt.bar(indices - width / 2, mean_hits_model, width, label="Model Mean Hits")
    plt.bar(
        indices + width / 2, mean_hits_wiener, width, label="Wiener Process Mean Hits"
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Hit Counts")
    plt.title("Mean Hit Counts Comparison")
    plt.xticks(
        indices, [f"{center[0]:.4f}" for center in non_zero_centers], rotation=45
    )
    plt.legend()

    # Plot for mean reversal counts comparison (excluding zero values)
    plt.subplot(2, 1, 2)
    plt.bar(
        indices - width / 2, mean_reversals_model, width, label="Model Mean Reversals"
    )
    plt.bar(
        indices + width / 2,
        mean_reversals_wiener,
        width,
        label="Wiener Process Mean Reversals",
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Reversal Counts")
    plt.title("Mean Reversal Counts Comparison")
    plt.xticks(
        indices, [f"{center[0]:.4f}" for center in non_zero_centers], rotation=45
    )
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Statistical comparison for hitrate (non-zero centers)
    non_zero_hits_model = np.array([hits[center[0]] for center in non_zero_centers])
    non_zero_wiener_hits = np.array(
        [wiener_hits[center[0]] for center in non_zero_centers]
    )
    t_stat_hits, p_value_hits = stats.ttest_ind(
        non_zero_hits_model, non_zero_wiener_hits
    )
    model_hit_rate = np.sum(non_zero_hits_model) / (
        len(test_prices) * len(non_zero_centers)
    )
    wiener_hit_rate = np.sum(non_zero_wiener_hits) / (
        len(wiener_process) * len(non_zero_centers)
    )
    print("Model Hit Rate:", model_hit_rate)
    print("Wiener Process Hit Rate:", wiener_hit_rate)
    print("Hit Rate t-statistic:", t_stat_hits)
    print("Hit Rate p-value:", p_value_hits)

    # Statistical comparison for reversals (non-zero centers)
    non_zero_actual_reversals = np.array(
        [actual_reversals[center[0]] for center in non_zero_centers]
    )
    non_zero_wiener_reversals = np.array(
        [wiener_reversals[center[0]] for center in non_zero_centers]
    )
    t_stat_reversals, p_value_reversals = stats.ttest_ind(
        non_zero_actual_reversals, non_zero_wiener_reversals
    )
    model_reversal_rate = np.sum(non_zero_actual_reversals) / (
        len(test_prices) * len(non_zero_centers)
    )
    wiener_reversal_rate = np.sum(non_zero_wiener_reversals) / (
        len(wiener_process) * len(non_zero_centers)
    )
    print("\n\nModel Reversal Rate:", model_reversal_rate)
    print("Wiener Process Reversal Rate:", wiener_reversal_rate)
    print("Reversal Rate t-statistic:", t_stat_reversals)
    print("Reversal Rate p-value:", p_value_reversals)


def ward(data_30m, data_5m, clusters):
    def calculate_reversals(prices, centers, look_forward=4, reversal_threshold=0.002):
        reversals = {center: 0 for center in centers}
        for i in range(len(prices)):
            for center in centers:
                if abs(prices[i] - center) / center <= margin_of_error:
                    for j in range(i + 1, min(i + 1 + look_forward, len(prices))):
                        if abs(prices[j] - prices[i]) / prices[i] >= reversal_threshold:
                            reversals[center] += 1
                            break
        return reversals

    # Load 30-minute data and split
    prices = data_30m["close"].values.reshape(-1, 1)
    train_prices, test_prices_30m = train_test_split(
        prices, test_size=0.075, shuffle=False
    )
    test_prices = data_5m[data_5m["time"] >= 1696867200]["close"].values

    # Applying Ward's hierarchical clustering
    ward = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
    labels = ward.fit_predict(
        data_30m[data_30m["time"] < 1696867200][["close", "open", "high", "low"]].values
    )

    # Calculate the mean price for each cluster
    centers = [
        prices[data_30m["time"] < 1696867200][labels == i].mean()
        for i in range(ward.n_clusters)
    ]

    # Define margin of error
    margin_of_error = 0.003  # ±0.3%

    # Calculate hits on the test set
    hits = {center: 0 for center in centers}
    for price in test_prices:
        for center in centers:
            if abs(price - center) / center <= margin_of_error:
                hits[center] += 1

    num_simulations = 100
    total_hits_model, total_reversals_model = np.zeros(len(centers)), np.zeros(
        len(centers)
    )
    total_hits_wiener, total_reversals_wiener = np.zeros(len(centers)), np.zeros(
        len(centers)
    )

    for seed in range(num_simulations):
        np.random.seed(seed)

        # Re-simulate Wiener process
        wiener_process = np.empty_like(test_prices)
        wiener_process[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_process[i] = wiener_process[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )

        # Calculate hits and reversals for actual data
        hits = {center: 0 for center in centers}
        actual_reversals = calculate_reversals(test_prices, centers)
        for price in test_prices:
            for center in centers:
                if abs(price - center) / center <= margin_of_error:
                    hits[center] += 1

        # Calculate hits and reversals for Wiener process
        wiener_hits = {center: 0 for center in centers}
        wiener_reversals = calculate_reversals(wiener_process, centers)
        for price in wiener_process:
            for center in centers:
                if abs(price - center) / center <= margin_of_error:
                    wiener_hits[center] += 1

        # Accumulate hits and reversals
        for i, center in enumerate(centers):
            total_hits_model[i] += hits[center]
            total_reversals_model[i] += actual_reversals[center]
            total_hits_wiener[i] += wiener_hits[center]
            total_reversals_wiener[i] += wiener_reversals[center]

    # Calculate mean hit and reversal rates, excluding levels with zero hits or reversals
    mean_hits_model = total_hits_model / num_simulations
    mean_reversals_model = total_reversals_model / num_simulations
    mean_hits_wiener = total_hits_wiener / num_simulations
    mean_reversals_wiener = total_reversals_wiener / num_simulations

    # Exclude levels with zero total hits or total reversals from mean values
    non_zero_indices_hits = np.where(mean_hits_model != 0)[0]
    non_zero_indices_reversals = np.where(mean_reversals_model != 0)[0]

    # Extract non-zero centers
    non_zero_centers_hits = [centers[i] for i in non_zero_indices_hits]
    non_zero_centers_reversals = [centers[i] for i in non_zero_indices_reversals]

    # Plotting with mean hit and reversal rates (excluding zero values)
    plt.figure(figsize=(15, 8))

    # Plot train data, test data, and the first 10 Wiener processes
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_prices)), train_prices, label="Train Data")
    plt.plot(
        range(len(train_prices), len(train_prices) + len(test_prices)),
        test_prices,
        label="Test Data",
    )
    for seed in range(10):
        np.random.seed(seed)
        wiener_temp = np.empty_like(test_prices)
        wiener_temp[0] = test_prices[0]
        for i in range(1, len(test_prices)):
            wiener_temp[i] = wiener_temp[i - 1] + np.random.normal(
                0, np.std(train_prices) / np.sqrt(len(train_prices))
            )
        plt.plot(
            range(len(train_prices), len(train_prices) + len(wiener_temp)),
            wiener_temp,
            alpha=0.15,
        )
    for level in centers:
        plt.axhline(y=level, color="red", linestyle="--")
    plt.title("Train Data, Test Data, and Wiener Processes")
    plt.legend()

    # Plot for mean hit counts comparison (excluding zero values)
    plt.subplot(2, 2, 2)
    width = 0.35
    indices_hits = np.arange(len(non_zero_centers_hits))
    plt.bar(
        indices_hits - width / 2,
        mean_hits_model[non_zero_indices_hits],
        width,
        label="Model Mean Hits",
    )
    plt.bar(
        indices_hits + width / 2,
        mean_hits_wiener[non_zero_indices_hits],
        width,
        label="Wiener Process Mean Hits",
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Hit Counts")
    plt.title("Mean Hit Counts Comparison")
    plt.xticks(
        indices_hits, [f"{center:.4f}" for center in non_zero_centers_hits], rotation=45
    )
    plt.legend()

    # Plot for mean reversal counts comparison (excluding zero values)
    plt.subplot(2, 1, 2)
    indices_reversals = np.arange(len(non_zero_centers_reversals))
    plt.bar(
        indices_reversals - width / 2,
        mean_reversals_model[non_zero_indices_reversals],
        width,
        label="Model Mean Reversals",
    )
    plt.bar(
        indices_reversals + width / 2,
        mean_reversals_wiener[non_zero_indices_reversals],
        width,
        label="Wiener Process Mean Reversals",
    )
    plt.xlabel("Cluster Centers")
    plt.ylabel("Mean Reversal Counts")
    plt.title("Mean Reversal Counts Comparison")
    plt.xticks(
        indices_reversals,
        [f"{center:.4f}" for center in non_zero_centers_reversals],
        rotation=45,
    )
    plt.legend()

    # Statistical comparison for hit rate (excluding zero values)
    non_zero_hits_model = np.array(list(hits.values()))[non_zero_indices_hits]
    non_zero_wiener_hits = np.array(list(wiener_hits.values()))[non_zero_indices_hits]
    t_stat, p_value = stats.ttest_ind(non_zero_hits_model, non_zero_wiener_hits)
    print(
        "Model Hit Rate:",
        sum(non_zero_hits_model) / (len(test_prices) * len(non_zero_centers_hits)),
    )
    print(
        "Wiener Process Hit Rate:",
        sum(non_zero_wiener_hits) / (len(wiener_process) * len(non_zero_centers_hits)),
    )
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    # Statistical comparison for reversal rate (excluding zero values)
    non_zero_actual_reversals = np.array(list(actual_reversals.values()))[
        non_zero_indices_reversals
    ]
    non_zero_wiener_reversals = np.array(list(wiener_reversals.values()))[
        non_zero_indices_reversals
    ]
    t_stat_reversals, p_value_reversals = stats.ttest_ind(
        non_zero_actual_reversals, non_zero_wiener_reversals
    )
    print(
        "\n\nModel Reversal Rate:",
        sum(non_zero_actual_reversals)
        / (len(test_prices) * len(non_zero_centers_reversals)),
    )
    print(
        "Wiener Process Reversal Rate:",
        sum(non_zero_wiener_reversals)
        / (len(wiener_process) * len(non_zero_centers_reversals)),
    )
    print("Reversal t-statistic:", t_stat_reversals)
    print("Reversal p-value:", p_value_reversals)

    plt.tight_layout()
    plt.show()
