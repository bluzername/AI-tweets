"""Configuration management for the podcast to tweets pipeline."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class PodcastConfig:
    """Configuration for a podcast source."""
    name: str
    rss_url: str
    x_handle: Optional[str] = None
    creator_handles: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rss_url": self.rss_url,
            "x_handle": self.x_handle,
            "creator_handles": self.creator_handles,
            "categories": self.categories
        }


@dataclass
class AccountConfig:
    """Configuration for an X.com account."""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    bearer_token: Optional[str] = None
    target_audience: str = "General tech enthusiasts"
    tone: str = "casual"
    emoji_usage: str = "minimal"
    hashtag_style: str = "minimal"
    signature_phrases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_audience": self.target_audience,
            "tone": self.tone,
            "emoji_usage": self.emoji_usage,
            "hashtag_style": self.hashtag_style,
            "signature_phrases": self.signature_phrases
        }
    
    def get_credentials(self) -> Dict[str, str]:
        """Get X.com API credentials."""
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "access_token": self.access_token,
            "access_token_secret": self.access_token_secret,
            "bearer_token": self.bearer_token
        }


class Config:
    """Main configuration manager."""
    
    def __init__(self, config_file: str = "config.json", env_file: str = ".env"):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON configuration file
            env_file: Path to .env file
        """
        self.config_file = Path(config_file)
        self.env_file = Path(env_file)
        
        load_dotenv(self.env_file)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.whisper_model = os.getenv("WHISPER_MODEL", "whisper-1")
        self.gpt_model = os.getenv("GPT_MODEL", "gpt-4-turbo-preview")
        
        self.podcasts: List[PodcastConfig] = []
        self.accounts: Dict[str, AccountConfig] = {}
        
        self.days_back = int(os.getenv("DAYS_BACK", "7"))
        self.max_highlights = int(os.getenv("MAX_HIGHLIGHTS", "3"))
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}")
            self._create_default_config()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            for podcast_data in data.get("podcasts", []):
                self.podcasts.append(PodcastConfig(**podcast_data))
            
            for account_data in data.get("accounts", []):
                account = AccountConfig(
                    name=account_data["name"],
                    api_key=os.getenv(f"{account_data['name'].upper()}_API_KEY"),
                    api_secret=os.getenv(f"{account_data['name'].upper()}_API_SECRET"),
                    access_token=os.getenv(f"{account_data['name'].upper()}_ACCESS_TOKEN"),
                    access_token_secret=os.getenv(f"{account_data['name'].upper()}_ACCESS_TOKEN_SECRET"),
                    bearer_token=os.getenv(f"{account_data['name'].upper()}_BEARER_TOKEN"),
                    target_audience=account_data.get("target_audience", "General audience"),
                    tone=account_data.get("tone", "casual"),
                    emoji_usage=account_data.get("emoji_usage", "minimal"),
                    hashtag_style=account_data.get("hashtag_style", "minimal"),
                    signature_phrases=account_data.get("signature_phrases", [])
                )
                self.accounts[account.name] = account
            
            logger.info(f"Loaded {len(self.podcasts)} podcasts and {len(self.accounts)} accounts")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration file."""
        default_config = {
            "podcasts": [
                {
                    "name": "The Tim Ferriss Show",
                    "rss_url": "https://rss.art19.com/tim-ferriss-show",
                    "x_handle": "tferriss",
                    "creator_handles": ["tferriss"],
                    "categories": ["productivity", "entrepreneurship", "self-improvement"]
                },
                {
                    "name": "Lex Fridman Podcast",
                    "rss_url": "https://lexfridman.com/feed/podcast/",
                    "x_handle": "lexfridman",
                    "creator_handles": ["lexfridman"],
                    "categories": ["AI", "science", "philosophy"]
                }
            ],
            "accounts": [
                {
                    "name": "main",
                    "target_audience": "Tech professionals and entrepreneurs",
                    "tone": "professional",
                    "emoji_usage": "minimal",
                    "hashtag_style": "trending",
                    "signature_phrases": ["Let's dive in", "Key takeaway"]
                },
                {
                    "name": "casual",
                    "target_audience": "Young professionals interested in self-improvement",
                    "tone": "casual",
                    "emoji_usage": "moderate",
                    "hashtag_style": "minimal",
                    "signature_phrases": ["Mind = blown", "This hit different"]
                }
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default config at {self.config_file}")
        
        self._create_env_example()
    
    def _create_env_example(self):
        """Create .env.example file."""
        if self.env_file.with_suffix('.example').exists():
            return
        
        env_example = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_MODEL=whisper-1
GPT_MODEL=gpt-4-turbo-preview

# X.com API Credentials for Main Account
MAIN_API_KEY=your_x_api_key
MAIN_API_SECRET=your_x_api_secret
MAIN_ACCESS_TOKEN=your_x_access_token
MAIN_ACCESS_TOKEN_SECRET=your_x_access_token_secret
MAIN_BEARER_TOKEN=your_x_bearer_token

# X.com API Credentials for Casual Account
CASUAL_API_KEY=your_x_api_key
CASUAL_API_SECRET=your_x_api_secret
CASUAL_ACCESS_TOKEN=your_x_access_token
CASUAL_ACCESS_TOKEN_SECRET=your_x_access_token_secret
CASUAL_BEARER_TOKEN=your_x_bearer_token

# Pipeline Configuration
DAYS_BACK=7
MAX_HIGHLIGHTS=3
DRY_RUN=false

# Logging
LOG_LEVEL=INFO
"""
        
        with open(self.env_file.with_suffix('.example'), 'w') as f:
            f.write(env_example)
        
        logger.info(f"Created .env.example file")
    
    def get_podcast_feeds(self) -> List[str]:
        """Get list of podcast RSS feed URLs."""
        return [podcast.rss_url for podcast in self.podcasts]
    
    def get_podcast_by_name(self, name: str) -> Optional[PodcastConfig]:
        """Get podcast configuration by name."""
        for podcast in self.podcasts:
            if podcast.name.lower() == name.lower():
                return podcast
        return None
    
    def get_account_config(self, account_name: str) -> Optional[AccountConfig]:
        """Get account configuration."""
        return self.accounts.get(account_name)
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openai_api_key:
            logger.error("OpenAI API key not configured")
            return False
        
        if not self.podcasts:
            logger.error("No podcasts configured")
            return False
        
        if not self.accounts:
            logger.error("No X.com accounts configured")
            return False
        
        return True
    
    def save(self):
        """Save current configuration to file."""
        data = {
            "podcasts": [p.to_dict() for p in self.podcasts],
            "accounts": [a.to_dict() for a in self.accounts.values()]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to {self.config_file}")