"""
Setup script for the Coin Futures Trading Bot.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="coin-futures-bot",
    version="1.0.0",
    author="AI Systems Architect",
    author_email="ai@tradingbot.com",
    description="AI-powered algorithmic trading system for cryptocurrency futures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/coin-futures-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
        ],
        "ml": [
            "optuna>=3.4.0",
            "mlflow>=2.8.1",
            "stable-baselines3>=2.2.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-bot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.sql"],
    },
)
