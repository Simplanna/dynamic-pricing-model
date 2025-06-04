# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dynamic pricing engine for cannabis dispensaries in Rhode Island and Massachusetts. The system optimizes prices based on:
- Product age
- Inventory levels
- Demand patterns
- Competition analysis
- Compliance constraints

## Development Environment

- **Python Version**: 3.9
- **Virtual Environment**: `.venv` directory present
- **IDE**: IntelliJ IDEA/PyCharm (based on `.idea` configuration)

## Development Guidelines

1. **Code Structure**: All Python code must be modular and readable
2. **Data Libraries**: Use Pandas, NumPy, and Requests for data handling
3. **Project Organization**: Store core logic in a `/pricing_engine` folder
4. **UI Framework**: Use Streamlit only for the UI
5. **Protected Files**: Do not modify `.env` or `credentials.py` files
6. **Architecture Changes**: Always ask before making large architectural changes
7. **Helper Files**: Use `claude_tasks/` directory to drop task-specific helper files

## Common Development Tasks

- **Install dependencies**: `pip install -r requirements.txt` (once created)
- **Activate virtual environment**: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

## Architecture Notes

The pricing engine should be structured with clear separation between:
- Data processing layer (using Pandas/NumPy)
- Pricing algorithm/model logic
- API integrations
- Streamlit UI layer

## APIs and Integrations

- **ScrapingBee Scraping Engine**: Key = 99XUHEZ6W021CVZWRCOO6KZ016HMFL3XFH43FB7AU556GZ94DA0RGA137PBLLYW1QSOOAXCVPPV2W930 (internal)
- **State Compliance APIs**: Optional integration for RI/MA compliance
- **POS Inventory Webhook**: Simulation (mocked for development)