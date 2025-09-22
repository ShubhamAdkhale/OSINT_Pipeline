import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
from collections import defaultdict
import json
import os

class DataVisualizer:
    def __init__(self, save_dir="visualizations"):
        """
        Initialize DataVisualizer with platform-specific settings
        
        Args:
            save_dir (str): Directory to save visualization files
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Platform-specific color scheme
        self.platform_colors = {
            'twitter': {
                'primary': '#1DA1F2',
                'secondary': '#14171A',
                'positive': '#17BF63',
                'negative': '#E0245E',
                'neutral': '#657786'
            },
            'reddit': {
                'primary': '#FF4500',
                'secondary': '#1A1A1B',
                'positive': '#46D160',
                'negative': '#EA0027',
                'neutral': '#7C7C7C'
            },
            'github': {
                'primary': '#24292E',
                'secondary': '#586069',
                'positive': '#28A745',
                'negative': '#D73A49',
                'neutral': '#6A737D'
            }
        }
        
        # Platform-specific metrics
        self.platform_metrics = {
            'twitter': ['retweets', 'likes', 'replies', 'quotes'],
            'reddit': ['score', 'upvote_ratio', 'comments'],
            'github': ['stars', 'forks', 'issues', 'prs']
        }
        
        # Custom stopwords for different contexts
        self.custom_stopwords = set(STOPWORDS).union({
            'http', 'https', 'www', 'com', 'rt', 'amp',
            'twitter', 'tweet', 'reddit', 'github', 'git',
            'user', 'users', 'said', 'say', 'says', 'one',
            'like', 'just', 'good', 'time', 'make', 'know'
        })
    
def generate_test_data():
    """Generate sample data for testing visualizations."""
    platforms = ['twitter', 'reddit', 'github']
    dates = pd.date_range(start='2025-01-01', end='2025-09-22', freq='D')
    
    platform_data = {}
    for platform in platforms:
        # Generate sentiment data
        sentiment_data = {
            'positive': {'score': np.random.uniform(0.3, 0.5), 'confidence': np.random.uniform(0.7, 0.9)},
            'neutral': {'score': np.random.uniform(0.2, 0.4), 'confidence': np.random.uniform(0.6, 0.8)},
            'negative': {'score': np.random.uniform(0.1, 0.3), 'confidence': np.random.uniform(0.7, 0.9)}
        }
        
        # Generate engagement data
        engagement_data = {
            'posts': np.random.randint(100, 1000, size=len(dates)),
            'users': np.random.randint(50, 500, size=len(dates)),
            'interactions': np.random.randint(500, 5000, size=len(dates))
        }
        
        # Generate time series data
        time_data = {
            'dates': dates,
            'values': np.cumsum(np.random.normal(0, 1, size=len(dates)))
        }
        
        platform_data[platform] = {
            'sentiment': sentiment_data,
            'engagement': engagement_data,
            'time_data': time_data
        }
    
    # Generate interaction network data
    interactions = {
        'nodes': [f'user_{i}' for i in range(20)],
        'edges': [(f'user_{i}', f'user_{j}', np.random.randint(1, 10))
                 for i in range(20) for j in range(i+1, 20) 
                 if np.random.random() < 0.2]
    }
    
    # Generate content pattern data
    content_patterns = {
        'topics': ['tech', 'politics', 'science', 'sports'],
        'frequencies': np.random.randint(10, 100, size=4),
        'sentiments': np.random.uniform(-1, 1, size=4)
    }
    
    return platform_data, interactions, content_patterns

def test_visualizer():
    """Test the DataVisualizer class with sample data"""
    visualizer = DataVisualizer()
    platform_data, interactions, content_patterns = generate_test_data()
    
    print("\nTesting Platform-Specific Visualizations...")
    
    # Test sentiment distribution for each platform
    for platform, data in platform_data.items():
        print(f"\n1. Testing {platform.title()} Sentiment Distribution")
        visualizer.plot_sentiment_distribution(
            data['sentiment'],
            platform=platform,
            save_name=f"{platform}_sentiment.html"
        )
    
    # Test sentiment trends with confidence
    for platform, data in platform_data.items():
        print(f"\n2. Testing {platform.title()} Sentiment Trends")
        df = pd.DataFrame({
            'compound': np.random.uniform(-1, 1, len(data['time_data']['dates']))
        }, index=data['time_data']['dates'])
        confidence_data = pd.DataFrame({
            'confidence': np.random.uniform(0.6, 1.0, len(data['time_data']['dates']))
        }, index=data['time_data']['dates'])
        visualizer.plot_sentiment_trends(
            df,
            platform=platform,
            confidence_data=confidence_data,
            save_name=f"{platform}_trends.html"
        )
    
    # Test engagement visualization
    print("\n3. Testing Platform Engagement Analysis")
    platform_engagement = {
        platform: data['engagement']
        for platform, data in platform_data.items()
    }
    platform_engagement['time_data'] = {
        platform: data['time_data']
        for platform, data in platform_data.items()
    }
    visualizer.plot_engagement_by_platform(
        platform_engagement,
        time_series=True,
        save_name="engagement_analysis.html"
    )
    
    # Test interaction network
    print("\n4. Testing User Interaction Network")
    for platform in platform_data.keys():
        visualizer.plot_interaction_network(
            interactions,
            platform=platform,
            save_name=f"{platform}_network.html"
        )
    
    # Test content patterns
    print("\n5. Testing Content Pattern Analysis")
    for platform in platform_data.keys():
        visualizer.plot_content_patterns(
            content_patterns,
            platform=platform,
            save_name=f"{platform}_patterns.html"
        )
    
    # Test comprehensive dashboard
    print("\n6. Testing Platform-Specific Dashboard")
    dashboard_data = {
        platform: {
            'sentiment': data['sentiment'],
            'engagement': data['engagement'],
            'time_data': data['time_data'],
            'interactions': interactions,
            'patterns': content_patterns
        }
        for platform, data in platform_data.items()
    }
    visualizer.create_dashboard(dashboard_data, save_name="platform_dashboard.html")
    
    print("\nVisualization tests complete! Check the 'visualizations' directory for output files.")

if __name__ == "__main__":
    test_visualizer()

    def plot_sentiment_distribution(self, sentiment_data, platform=None, 
                                  title="Sentiment Distribution", save_name="sentiment_dist.html"):
        """
        Create an interactive sentiment distribution visualization
        
        Args:
            sentiment_data (dict): Dictionary containing sentiment distribution
            platform (str): Platform name for color scheme
            title (str): Chart title
            save_name (str): Filename to save the plot
        """
        try:
            # Get platform-specific colors
            colors = None
            if platform and platform in self.platform_colors:
                colors = [
                    self.platform_colors[platform]['positive'],
                    self.platform_colors[platform]['neutral'],
                    self.platform_colors[platform]['negative']
                ]
            else:
                colors = ['#2ecc71', '#95a5a6', '#e74c3c']

            # Process sentiment data by confidence
            sentiment_df = pd.DataFrame([
                {
                    'Sentiment': sent_type,
                    'Score': score,
                    'Confidence': sent_data.get('confidence', 1.0)
                }
                for sent_type, sent_data in sentiment_data.items()
                for score in [sent_data.get('score', 0) if isinstance(sent_data, dict) else sent_data]
            ])

            # Create sunburst chart for hierarchical sentiment view
            fig = go.Figure(go.Sunburst(
                labels=sentiment_df['Sentiment'],
                parents=[''] * len(sentiment_df),
                values=sentiment_df['Score'],
                marker=dict(colors=colors),
                hovertemplate="""
                Sentiment: %{label}<br>
                Score: %{value:.1f}%<br>
                Confidence: %{color:.2f}
                <extra></extra>
                """
            ))

            # Update layout with platform context
            fig.update_layout(
                title=title,
                title_x=0.5,
                width=800,
                height=800,
                showlegend=False
            )

            # Add platform-specific annotations
            if platform:
                fig.add_annotation(
                    text=f"Platform: {platform.title()}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.1,
                    showarrow=False
                )

            # Save interactive plot
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating sentiment distribution plot: {str(e)}")

    def plot_sentiment_trends(self, df, platform=None, confidence_data=None,
                            title="Sentiment Trends Over Time", save_name="sentiment_trends.html"):
        """
        Create an interactive line plot of sentiment trends with confidence bands
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index and sentiment scores
            platform (str): Platform name for styling
            confidence_data (pd.DataFrame): Optional confidence scores
            title (str): Plot title
            save_name (str): Filename to save the plot
        """
        try:
            # Create figure with secondary y-axis for confidence
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Get platform colors
            colors = self.platform_colors.get(platform, {
                'primary': '#2980b9',
                'secondary': '#95a5a6'
            })

            # Add main sentiment trend
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['compound'],
                    name="Sentiment",
                    line=dict(color=colors['primary'], width=2),
                    mode='lines'
                )
            )

            # Add rolling mean
            rolling_mean = df['compound'].rolling(window=7).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_mean,
                    name="7-day Average",
                    line=dict(color=colors['secondary'], width=2, dash='dash'),
                    mode='lines'
                )
            )

            # Add confidence band if available
            if confidence_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=confidence_data['confidence'],
                        name="Confidence",
                        line=dict(color=colors.get('neutral', '#7f8c8d')),
                        mode='lines',
                        yaxis="y2"
                    ),
                    secondary_y=True
                )

            # Update layout
            fig.update_layout(
                title=title + (f" ({platform.title()})" if platform else ""),
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                yaxis2_title="Confidence",
                hovermode='x unified',
                template='plotly_white',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)

            # Update y-axes ranges
            fig.update_yaxes(range=[-1, 1], secondary_y=False)
            if confidence_data is not None:
                fig.update_yaxes(range=[0, 1], secondary_y=True)

            # Save interactive plot
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating sentiment trends plot: {str(e)}")

    def create_wordcloud(self, texts, platform=None, sentiment_filter=None, 
                        title="Word Cloud", save_name="wordcloud.png"):
        """
        Create a platform-specific word cloud with sentiment filtering
        
        Args:
            texts (list): List of text strings or dicts with text and sentiment
            platform (str): Platform name for color scheme
            sentiment_filter (str): Filter for 'positive', 'negative', or 'neutral' text
            title (str): Plot title
            save_name (str): Filename to save the plot
        """
        try:
            # Filter texts by sentiment if needed
            if sentiment_filter and isinstance(texts[0], dict):
                texts = [
                    t['text'] for t in texts 
                    if t.get('sentiment_label', '').startswith(sentiment_filter)
                ]
            elif not isinstance(texts[0], dict):
                texts = [str(t) for t in texts]
            
            # Combine all texts
            text = " ".join(texts)
            
            # Get platform-specific colors and stopwords
            colors = None
            if platform and platform in self.platform_colors:
                colors = self.platform_colors[platform]
                
            # Add platform-specific stopwords
            stopwords = self.custom_stopwords.copy()
            if platform:
                stopwords.update({platform, f"{platform}_user", f"{platform}_post"})
            
            # Create color function based on platform
            if colors:
                color_func = lambda word, font_size, position, orientation, random_state=None, **kwargs: \
                    colors.get('primary' if random_state.rand() < 0.7 else 'secondary', '#000000')
            else:
                color_func = None
            
            # Create and generate word cloud
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                stopwords=stopwords,
                color_func=color_func,
                max_words=150,
                prefer_horizontal=0.7,
                collocations=True,
                min_word_length=3
            ).generate(text)
            
            # Create figure with platform-specific styling
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Add title with platform context
            full_title = title
            if platform:
                full_title += f" ({platform.title()})"
            if sentiment_filter:
                full_title += f" - {sentiment_filter.title()} Sentiment"
            plt.title(full_title, size=16, pad=20)
            
            # Save high-resolution plot
            plt.savefig(
                os.path.join(self.save_dir, save_name),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            print(f"Error creating word cloud: {str(e)}")

    def plot_engagement_by_platform(self, data, time_series=False,
                                  title="Engagement by Platform",
                                  save_name="platform_engagement.html"):
        """
        Create an interactive visualization of engagement metrics by platform
        
        Args:
            data (dict): Platform engagement data with detailed metrics
            time_series (bool): Whether to show time-series data
            title (str): Plot title
            save_name (str): Filename to save the plot
        """
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Overall Engagement",
                    "Engagement Distribution",
                    "Engagement by Type",
                    "Engagement Trends"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # Process data for each platform
            platforms_processed = {}
            for platform, metrics in data.items():
                if isinstance(metrics, dict):
                    # Get platform colors
                    colors = self.platform_colors.get(platform, {
                        'primary': '#2980b9',
                        'secondary': '#95a5a6'
                    })
                    
                    # Calculate total engagement
                    total = sum(
                        v for k, v in metrics.items() 
                        if k in self.platform_metrics.get(platform, [])
                    )
                    
                    # Store processed data
                    platforms_processed[platform] = {
                        'total': total,
                        'metrics': metrics,
                        'colors': colors
                    }
            
            # 1. Overall Engagement Bar Chart
            fig.add_trace(
                go.Bar(
                    x=list(platforms_processed.keys()),
                    y=[p['total'] for p in platforms_processed.values()],
                    marker_color=[p['colors']['primary'] for p in platforms_processed.values()],
                    name="Total Engagement"
                ),
                row=1, col=1
            )
            
            # 2. Engagement Distribution Pie Chart
            fig.add_trace(
                go.Pie(
                    labels=list(platforms_processed.keys()),
                    values=[p['total'] for p in platforms_processed.values()],
                    marker_colors=[p['colors']['primary'] for p in platforms_processed.values()]
                ),
                row=1, col=2
            )
            
            # 3. Engagement by Type Stacked Bar
            for platform in platforms_processed:
                metrics = platforms_processed[platform]['metrics']
                for metric in self.platform_metrics.get(platform, []):
                    if metric in metrics:
                        fig.add_trace(
                            go.Bar(
                                name=f"{platform}-{metric}",
                                x=[platform],
                                y=[metrics[metric]],
                                marker_color=platforms_processed[platform]['colors']['secondary']
                            ),
                            row=2, col=1
                        )
            
            # 4. Time Series if available
            if time_series and 'time_data' in data:
                for platform in platforms_processed:
                    if platform in data['time_data']:
                        fig.add_trace(
                            go.Scatter(
                                x=data['time_data'][platform]['dates'],
                                y=data['time_data'][platform]['values'],
                                name=platform,
                                line=dict(color=platforms_processed[platform]['colors']['primary'])
                            ),
                            row=2, col=2
                        )
            
            # Update layout
            fig.update_layout(
                title_text=title,
                showlegend=True,
                height=1000,
                template='plotly_white',
                barmode='stack'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Platform", row=1, col=1)
            fig.update_yaxes(title_text="Total Engagement", row=1, col=1)
            fig.update_xaxes(title_text="Platform", row=2, col=1)
            fig.update_yaxes(title_text="Engagement by Type", row=2, col=1)
            if time_series:
                fig.update_xaxes(title_text="Date", row=2, col=2)
                fig.update_yaxes(title_text="Engagement", row=2, col=2)
            
            # Save interactive plot
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating platform engagement plot: {str(e)}")

    def plot_hashtag_trends(self, hashtag_data, title="Top Hashtags",
                           save_name="hashtag_trends.png", top_n=10):
        """
        Create a horizontal bar chart of top hashtags
        Args:
            hashtag_data (dict): Dictionary of hashtag frequencies
            title (str): Plot title
            save_name (str): Filename to save the plot
            top_n (int): Number of top hashtags to show
        """
        try:
            # Sort and get top N hashtags
            sorted_data = dict(sorted(hashtag_data.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:top_n])
            
            # Create horizontal bar chart
            plt.figure(figsize=(12, 6))
            plt.barh(list(sorted_data.keys()), list(sorted_data.values()))
            plt.title(title)
            plt.xlabel('Frequency')
            plt.ylabel('Hashtags')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, save_name))
            plt.close()
            
        except Exception as e:
            print(f"Error creating hashtag trends plot: {str(e)}")

    def plot_user_activity(self, user_data, title="User Activity",
                          save_name="user_activity.html", top_n=10):
        """
        Create an interactive scatter plot of user activity
        Args:
            user_data (list): List of user dictionaries with activity metrics
            title (str): Plot title
            save_name (str): Filename to save the plot
            top_n (int): Number of top users to show
        """
        try:
            df = pd.DataFrame(user_data)
            df = df.nlargest(top_n, 'posts_count')
            
            fig = px.scatter(
                df,
                x='followers_count',
                y='engagement_rate',
                size='posts_count',
                text='username',
                title=title,
                labels={
                    'followers_count': 'Followers Count',
                    'engagement_rate': 'Engagement Rate',
                    'posts_count': 'Number of Posts'
                }
            )
            
            fig.update_traces(textposition='top center')
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating user activity plot: {str(e)}")

    def plot_interaction_network(self, interactions, platform=None, 
                               title="User Interaction Network", save_name="network.html"):
        """
        Create an interactive network visualization of user interactions
        
        Args:
            interactions (list): List of interaction records with source, target, and type
            platform (str): Platform name for styling
            title (str): Plot title
            save_name (str): Filename to save the plot
        """
        try:
            # Create network graph
            G = nx.Graph()
            
            # Process interactions
            edge_types = defaultdict(int)
            for interaction in interactions:
                source = interaction['source']
                target = interaction['target']
                int_type = interaction.get('type', 'interaction')
                
                # Add edge with weight
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1)
                
                edge_types[int_type] += 1
            
            # Calculate node metrics
            node_degrees = dict(G.degree())
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            
            # Get platform colors
            colors = self.platform_colors.get(platform, {
                'primary': '#2980b9',
                'secondary': '#95a5a6',
                'positive': '#27ae60',
                'negative': '#c0392b'
            })
            
            # Create plot
            pos = nx.spring_layout(G)
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=colors['secondary']),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    size=[v * 10 for v in eigenvector_centrality.values()],
                    color=list(node_degrees.values()),
                    colorscale='YlOrRd',
                    line_width=2
                ),
                text=[f"User: {node}<br>Connections: {node_degrees[node]}" for node in G.nodes()]
            ))
            
            # Update layout
            fig.update_layout(
                title=title + (f" ({platform.title()})" if platform else ""),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white'
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating interaction network: {str(e)}")
            
    def plot_content_patterns(self, content_data, platform=None,
                          title="Content Patterns Analysis", save_name="patterns.html"):
        """
        Visualize platform-specific content patterns
        
        Args:
            content_data (dict): Content analysis data including frequencies and patterns
            platform (str): Platform name for context
            title (str): Plot title
            save_name (str): Filename to save the plot
        """
        try:
            # Create subplots for different aspects
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Post Timing Distribution",
                    "Content Length Distribution",
                    "Common Patterns",
                    "Engagement by Content Type"
                )
            )
            
            # Get platform colors
            colors = self.platform_colors.get(platform, {
                'primary': '#2980b9',
                'secondary': '#95a5a6'
            })
            
            # 1. Post Timing Heatmap
            if 'timing' in content_data:
                timing_df = pd.DataFrame(content_data['timing'])
                fig.add_trace(
                    go.Heatmap(
                        z=timing_df.values,
                        x=timing_df.columns,
                        y=timing_df.index,
                        colorscale='Viridis'
                    ),
                    row=1, col=1
                )
            
            # 2. Content Length Distribution
            if 'lengths' in content_data:
                fig.add_trace(
                    go.Histogram(
                        x=content_data['lengths'],
                        nbinsx=30,
                        marker_color=colors['primary']
                    ),
                    row=1, col=2
                )
            
            # 3. Common Patterns
            if 'patterns' in content_data:
                patterns = pd.DataFrame(content_data['patterns'])
                fig.add_trace(
                    go.Bar(
                        x=patterns['pattern'],
                        y=patterns['frequency'],
                        marker_color=colors['primary']
                    ),
                    row=2, col=1
                )
            
            # 4. Engagement by Content Type
            if 'engagement' in content_data:
                engagement = pd.DataFrame(content_data['engagement'])
                fig.add_trace(
                    go.Bar(
                        x=engagement['type'],
                        y=engagement['value'],
                        marker_color=colors['primary']
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text=title + (f" ({platform.title()})" if platform else ""),
                height=1000,
                template='plotly_white',
                showlegend=False
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating content patterns plot: {str(e)}")
    
    def create_dashboard(self, data, save_name="dashboard.html"):
        """
        Create a comprehensive interactive dashboard with platform-specific insights
        
        Args:
            data (dict): Dictionary containing all visualization data
            save_name (str): Filename to save the dashboard
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Sentiment Distribution",
                    "Engagement by Platform",
                    "Sentiment Trends",
                    "Top Hashtags"
                )
            )
            
            # Add sentiment distribution pie chart
            fig.add_trace(
                go.Pie(
                    labels=['Positive', 'Neutral', 'Negative'],
                    values=[
                        data['sentiment'].get('positive', 0),
                        data['sentiment'].get('neutral', 0),
                        data['sentiment'].get('negative', 0)
                    ]
                ),
                row=1, col=1
            )
            
            # Add platform engagement bar chart
            fig.add_trace(
                go.Bar(
                    x=list(data['engagement'].keys()),
                    y=list(data['engagement'].values())
                ),
                row=1, col=2
            )
            
            # Add sentiment trends line chart
            fig.add_trace(
                go.Scatter(
                    x=data['trends'].index,
                    y=data['trends']['compound'],
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # Add hashtag trends bar chart
            sorted_hashtags = dict(sorted(
                data['hashtags'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            fig.add_trace(
                go.Bar(
                    x=list(sorted_hashtags.keys()),
                    y=list(sorted_hashtags.values())
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text="Social Media Analysis Dashboard"
            )
            
            # Save dashboard
            fig.write_html(os.path.join(self.save_dir, save_name))
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")

def test_visualizer():
    """Test the visualization functionality with platform-specific features"""
    visualizer = DataVisualizer()
    
    # Test data for each platform
    platform_data = {
        'twitter': {
            'sentiment': {
                'positive': {'score': 45, 'confidence': 0.85},
                'neutral': {'score': 30, 'confidence': 0.75},
                'negative': {'score': 25, 'confidence': 0.90}
            },
            'engagement': {
                'retweets': 500,
                'likes': 1200,
                'replies': 300,
                'quotes': 100
            },
            'time_data': {
                'dates': pd.date_range(start='2025-09-01', end='2025-09-22'),
                'values': np.random.uniform(100, 1000, 22)
            }
        },
        'reddit': {
            'sentiment': {
                'positive': {'score': 40, 'confidence': 0.80},
                'neutral': {'score': 35, 'confidence': 0.70},
                'negative': {'score': 25, 'confidence': 0.85}
            },
            'engagement': {
                'score': 1500,
                'upvote_ratio': 0.75,
                'comments': 800
            },
            'time_data': {
                'dates': pd.date_range(start='2025-09-01', end='2025-09-22'),
                'values': np.random.uniform(50, 800, 22)
            }
        },
        'github': {
            'sentiment': {
                'positive': {'score': 50, 'confidence': 0.75},
                'neutral': {'score': 40, 'confidence': 0.80},
                'negative': {'score': 10, 'confidence': 0.95}
            },
            'engagement': {
                'stars': 300,
                'forks': 150,
                'issues': 50,
                'prs': 25
            },
            'time_data': {
                'dates': pd.date_range(start='2025-09-01', end='2025-09-22'),
                'values': np.random.uniform(10, 200, 22)
            }
        }
    }
    
    # Sample interaction data
    interactions = [
        {'source': 'user1', 'target': 'user2', 'type': 'reply'},
        {'source': 'user1', 'target': 'user3', 'type': 'mention'},
        {'source': 'user2', 'target': 'user4', 'type': 'quote'},
        {'source': 'user3', 'target': 'user4', 'type': 'reply'},
        {'source': 'user2', 'target': 'user5', 'type': 'mention'},
    ]
    
    # Sample content patterns data
    content_patterns = {
        'timing': pd.DataFrame(
            np.random.randint(0, 100, size=(7, 24)),
            index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            columns=[f'{i:02d}:00' for i in range(24)]
        ),
        'lengths': np.random.normal(100, 30, 1000),
        'patterns': [
            {'pattern': 'URL', 'frequency': 150},
            {'pattern': 'Hashtag', 'frequency': 120},
            {'pattern': 'Mention', 'frequency': 100},
            {'pattern': 'Emoji', 'frequency': 80}
        ],
        'engagement': [
            {'type': 'Text', 'value': 100},
            {'type': 'Image', 'value': 150},
            {'type': 'Video', 'value': 200},
            {'type': 'Link', 'value': 80}
        ]
    }

if __name__ == "__main__":
    test_visualizer()
