# QuantumTrader AI v2.0 - API Documentation

## Overview

QuantumTrader AI provides a RESTful API for trading operations, portfolio management, and market data access. All endpoints return JSON responses and use standard HTTP status codes.

## Base URL

```
http://localhost:7000/api
```

## Authentication

The API uses session-based authentication. User sessions are automatically created and managed through browser cookies.

## Endpoints

### Market Data

#### Get Market Data
```http
GET /api/market_data
```

Returns real-time market data for all supported stocks.

**Response:**
```json
{
  "AAPL": {
    "symbol": "AAPL",
    "price": 228.50,
    "change": 1.25,
    "volume": 45000000,
    "high": 230.00,
    "low": 226.00,
    "rsi": 65.4
  },
  "GOOGL": {
    "symbol": "GOOGL",
    "price": 175.80,
    "change": -0.45,
    "volume": 22000000,
    "high": 177.00,
    "low": 174.50,
    "rsi": 58.2
  }
}
```

### Portfolio Management

#### Get Portfolio Information
```http
GET /api/portfolio/{portfolio_type}
```

**Parameters:**
- `portfolio_type` (string): Either "human" or "ai"

**Response:**
```json
{
  "cash": 95000.00,
  "total_value": 105000.00,
  "pnl": 5000.00,
  "positions": {
    "AAPL": {
      "quantity": 50,
      "avg_price": 220.00,
      "total_cost": 11000.00
    }
  },
  "enabled": true,
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "strategy": "conservative",
  "trades": 15
}
```

### Trading Operations

#### Validate Trade
```http
POST /api/validate_trade
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "quantity": 10,
  "price": 228.50,
  "action": "BUY"
}
```

**Response:**
```json
{
  "valid": true,
  "error": null
}
```

**Error Response:**
```json
{
  "valid": false,
  "error": "Недостаточно средств. Доступно: $5000.00, Требуется: $10000.00"
}
```

#### Execute Trade
```http
POST /api/execute_trade
```

**Request Body:**
```json
{
  "portfolio_type": "human",
  "action": "BUY",
  "symbol": "AAPL",
  "quantity": 10,
  "price": 228.50
}
```

**Response:**
```json
{
  "success": true,
  "message": "Сделка выполнена успешно",
  "error": null
}
```

**Error Response:**
```json
{
  "success": false,
  "message": null,
  "error": "Недостаточно средств для выполнения сделки"
}
```

### Trade History

#### Get Trade History
```http
GET /api/trade_history
```

**Response:**
```json
[
  {
    "trade_type": "BUY",
    "symbol": "AAPL",
    "quantity": 10,
    "price": 228.50,
    "total_amount": 2285.00,
    "pnl": 0.00,
    "notes": "Покупка 10 акций AAPL по $228.50",
    "timestamp": "2025-08-04T19:30:00"
  },
  {
    "trade_type": "SELL",
    "symbol": "AAPL",
    "quantity": 5,
    "price": 235.00,
    "total_amount": 1175.00,
    "pnl": 32.50,
    "notes": "Продажа 5 акций AAPL по $235.00, P&L: $32.50",
    "timestamp": "2025-08-04T20:15:00"
  }
]
```

### Analytics

#### Get Trading Analytics
```http
GET /api/analytics
```

**Response:**
```json
{
  "total_trades": 25,
  "successful_trades": 18,
  "win_rate": 72.0,
  "total_volume": 125000.00,
  "realized_pnl": 2500.00,
  "avg_trade_size": 5000.00
}
```

### AI Trading

#### Toggle AI Bot
```http
POST /api/ai/toggle
```

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "AI Bot включен",
  "enabled": true
}
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "error": "Error message description",
  "code": "ERROR_CODE",
  "details": "Additional error details"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, consider implementing rate limiting to prevent abuse.

## Data Types

### Stock Symbol
Valid stock symbols: `AAPL`, `GOOGL`, `MSFT`, `AMZN`, `TSLA`, `NVDA`, `META`, `NFLX`

### Portfolio Type
- `human` - Manual trading portfolio
- `ai` - AI-managed portfolio

### Trade Action
- `BUY` - Purchase stocks
- `SELL` - Sell stocks

### AI Provider
- `openai` - OpenAI GPT models
- `claude` - Anthropic Claude models
- `gemini` - Google Gemini models
- `local` - Local LLM models

### Trading Strategy
- `conservative` - Low-risk trading approach
- `aggressive` - High-risk, high-reward approach
- `balanced` - Moderate risk approach

## Trading Limits

### Security Constraints
- Maximum quantity per trade: 10,000 shares
- Maximum trade value: $50,000
- Minimum cash reserve: $1,000
- Supported stocks: 8 major US stocks

### Validation Rules
- Quantity must be positive
- Price must be positive
- Symbol must be from supported list
- Sufficient funds required for purchases
- Sufficient shares required for sales

## Market Data

### Data Sources
- Primary: Yahoo Finance API
- Fallback: Simulated data with realistic variations

### Update Frequency
- Real-time data: Updated every 60 seconds
- Fallback data: Updated every 120 seconds on errors

### Data Fields
- `price` - Current stock price
- `change` - Percentage change from previous close
- `volume` - Trading volume
- `high` - Day's high price
- `low` - Day's low price
- `rsi` - Relative Strength Index (14-period)

## Examples

### Complete Trading Flow

1. **Get market data**
   ```bash
   curl -X GET http://localhost:7000/api/market_data
   ```

2. **Validate trade**
   ```bash
   curl -X POST http://localhost:7000/api/validate_trade \
     -H "Content-Type: application/json" \
     -d '{"symbol":"AAPL","quantity":10,"price":228.50,"action":"BUY"}'
   ```

3. **Execute trade**
   ```bash
   curl -X POST http://localhost:7000/api/execute_trade \
     -H "Content-Type: application/json" \
     -d '{"portfolio_type":"human","action":"BUY","symbol":"AAPL","quantity":10,"price":228.50}'
   ```

4. **Check portfolio**
   ```bash
   curl -X GET http://localhost:7000/api/portfolio/human
   ```

5. **View trade history**
   ```bash
   curl -X GET http://localhost:7000/api/trade_history
   ```

### AI Trading Setup

1. **Enable AI bot**
   ```bash
   curl -X POST http://localhost:7000/api/ai/toggle \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

2. **Check AI portfolio**
   ```bash
   curl -X GET http://localhost:7000/api/portfolio/ai
   ```

## Security Considerations

### Input Validation
- All inputs are validated server-side
- SQL injection protection through parameterized queries
- XSS prevention through proper escaping

### Session Management
- Secure session cookies
- Session timeout handling
- User isolation between sessions

### Trading Security
- Trade validation before execution
- Insufficient funds prevention
- Position verification for sales
- Maximum trade limits enforcement

## Support

For API support and questions:
1. Check the main documentation
2. Review error messages carefully
3. Ensure proper request formatting
4. Verify session state and authentication

