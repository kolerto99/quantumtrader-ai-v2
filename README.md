# QuantumTrader AI v2.0

**Universal Multi-AI Trading Platform with Enhanced NexusLabs Design**

A sophisticated trading simulation platform featuring real-time market data, AI-powered analysis, and a stunning cosmic interface inspired by NexusLabs design principles.

## 🌟 Features

### 🎨 Enhanced Visual Design
- **Cosmic Background**: Multi-layered radial gradients with animated star fields
- **Animated Particles**: 120+ floating particles with neon effects
- **NexusLabs Colors**: Cyan (#00d4ff), Magenta (#ff0080), Green (#00ff88)
- **Visual Effects**: Shimmer animations, pulsing borders, gradient transitions
- **Professional Interface**: Clean, emoji-free design for serious trading

### 💼 Trading Features
- **Individual Portfolios**: Each user gets $100,000 virtual funds
- **AI Portfolio**: Additional $50,000 for AI-powered trading
- **Real-time Data**: Live market data via Yahoo Finance API
- **Trade Validation**: Complete security with limits and checks
- **Position Management**: Track open positions with P&L calculations
- **Trade History**: Comprehensive transaction logging

### 🤖 AI Integration
- **Multi-Provider Support**: OpenAI, Claude, Gemini, Local LLM
- **Market Analysis**: AI-powered trading recommendations
- **Risk Assessment**: Automated risk level evaluation
- **Strategy Selection**: Conservative, Aggressive, Balanced approaches

### 📊 Analytics & Reporting
- **Performance Metrics**: Win rate, total volume, realized P&L
- **Trade Analytics**: Success rate and average trade size
- **Real-time Updates**: Live portfolio value calculations
- **Market Indicators**: RSI, volume, price changes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantumtrader-ai-v2.0
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

5. **Run the application**
   ```bash
   python src/app.py
   ```

6. **Access the platform**
   Open your browser and navigate to `http://localhost:7000`

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for AI trading features
- `FLASK_ENV`: Set to `development` for debug mode

### Trading Limits
- Maximum quantity per trade: 10,000 shares
- Maximum trade value: $50,000
- Minimum cash reserve: $1,000

### Supported Stocks
- AAPL (Apple Inc.)
- GOOGL (Alphabet Inc.)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- TSLA (Tesla Inc.)
- NVDA (NVIDIA Corporation)
- META (Meta Platforms Inc.)
- NFLX (Netflix Inc.)

## 📱 Usage

### Manual Trading
1. Select a stock symbol from the dropdown
2. Enter quantity and price
3. Click BUY or SELL
4. Confirm the trade in the modal dialog

### AI Trading
1. Navigate to the "AI Bot" tab
2. Enable the AI trading bot
3. Select AI provider and strategy
4. Monitor AI recommendations and trades

### Portfolio Management
- View real-time portfolio value
- Track open positions with current P&L
- Monitor cash balance and total value
- Sell positions directly from the positions table

### Analytics
- Review trade history with detailed records
- Analyze performance metrics
- Track win rate and trading volume
- Monitor realized and unrealized P&L

## 🏗️ Architecture

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive trading interface
- **Animations**: CSS keyframes and transitions
- **Real-time Updates**: AJAX polling for live data

### Backend
- **Flask**: Python web framework
- **SQLite**: Local database for user data
- **Threading**: Background market data updates
- **API Integration**: Yahoo Finance for real market data

### Database Schema
- **Users**: User identification and session management
- **Portfolios**: Human and AI portfolio data
- **Positions**: Current stock holdings
- **Trade History**: Complete transaction records

## 🔒 Security Features

### Trade Validation
- Input sanitization and type checking
- Quantity and price range validation
- Sufficient funds verification
- Position availability checks

### User Isolation
- Session-based user identification
- Individual portfolio separation
- Secure database transactions
- No cross-user data access

## 🎯 API Endpoints

### Market Data
- `GET /api/market_data` - Current market prices
- `GET /api/portfolio/<type>` - Portfolio information

### Trading
- `POST /api/validate_trade` - Validate trade parameters
- `POST /api/execute_trade` - Execute validated trade

### Analytics
- `GET /api/trade_history` - User trade history
- `GET /api/analytics` - Performance analytics

### AI Features
- `POST /api/ai/toggle` - Enable/disable AI trading

## 🌐 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For support and questions, please open an issue in the repository.

## 🔄 Version History

### v2.0 (Current)
- Enhanced NexusLabs design with cosmic background
- Animated particles and visual effects
- Improved color scheme and typography
- Real-time market data integration
- Individual user portfolios
- AI trading capabilities
- Professional interface without emojis

### v1.0
- Basic trading interface
- Simple portfolio management
- Manual trading only
- Basic market data simulation

---

**Built with ❤️ using Flask, Python, and modern web technologies**

