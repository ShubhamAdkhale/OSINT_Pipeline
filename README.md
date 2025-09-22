# OSINT Pipeline

A Python-based OSINT (Open Source Intelligence) data collection and analysis pipeline that gathers information from various social media platforms and online sources.

## Features

- Multi-platform data collection (Facebook, Instagram, Reddit, Twitter)
- Sentiment analysis
- Data cleaning and preprocessing
- Database storage
- Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/osint_pipeline.git
cd osint_pipeline
```

2. Create a virtual environment and activate it:
```bash
python -m venv osint_env
# On Windows
.\osint_env\Scripts\activate
# On Unix or MacOS
source osint_env/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your API credentials:
```
FACEBOOK_ACCESS_TOKEN=your_token_here
INSTAGRAM_USERNAME=your_username
INSTAGRAM_PASSWORD=your_password
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
```

## Usage

Run the main pipeline:
```bash
python main.py
```

## Project Structure

```
osint_pipeline/
├── main.py                 # Main execution file
├── collectors/            # Data collection modules
│   ├── facebook_collector.py
│   ├── instagram_collector.py
│   ├── reddit_collector.py
│   ├── snscrape_collector.py
│   └── twitter_collector.py
├── data/                 # Data storage
│   └── osint.db
└── utils/               # Utility functions
    ├── cleaner.py      # Data cleaning utilities
    ├── database.py     # Database operations
    ├── sentiment.py    # Sentiment analysis
    └── visualizer.py   # Data visualization
```

## License

[MIT License](LICENSE)