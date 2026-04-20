from src.visualize import plot_rmse_vs_samples, plot_los_nlos_comparison

if __name__ == '__main__':
    plot_rmse_vs_samples()
    plot_los_nlos_comparison(num_samples=50)