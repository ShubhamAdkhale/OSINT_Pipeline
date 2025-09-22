import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import os

class DataVisualizer:
    def __init__(self, save_dir="visualizations"):
        """Initialize visualization settings"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('default')

    def plot_sentiment_by_platform(self, data):
        """
        Create a bar plot showing average sentiment scores by platform.
        Matches the example graph style exactly.
        """
        try:
            # Convert data to DataFrame if it's not already
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            # Extract sentiment scores
            df['sentiment_score'] = df['sentiment'].apply(
                lambda x: x.get('compound', 0) if isinstance(x, dict) else x
            )

            # Calculate average sentiment by platform
            avg_sentiment = df.groupby('platform')['sentiment_score'].mean().reset_index()

            # Sort platforms in the desired order
            platform_order = ['github', 'reddit', 'twitter', 'facebook']
            avg_sentiment = avg_sentiment.set_index('platform').reindex(platform_order).reset_index()

            # Create the plot with specific dimensions
            plt.figure(figsize=(10, 6))

            # Create bar plot with consistent blue color
            bars = plt.bar(avg_sentiment['platform'], avg_sentiment['sentiment_score'],
                          color='#1f77b4')  # Standard matplotlib blue

            # Customize the plot
            plt.title('Average Sentiment by Platform', pad=20)
            plt.xlabel('Platform')
            plt.ylabel('Sentiment Score')

            # Set y-axis limits and ticks to match example
            plt.ylim(-0.15, 0.25)
            plt.yticks(np.arange(-0.15, 0.26, 0.05))

            # Add gridlines only for y-axis
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Save the plot
            save_path = os.path.join(self.save_dir, 'sentiment_by_platform.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Sentiment visualization saved to '{save_path}'")

            return avg_sentiment

        except Exception as e:
            print(f"⚠️ Error generating visualization: {str(e)}")
            raise e