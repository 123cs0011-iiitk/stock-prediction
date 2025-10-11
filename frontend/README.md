# Stock Price Insight Arena - Next.js Frontend

A modern, responsive stock price insight arena built with Next.js 14, React, TypeScript, and Tailwind CSS. This frontend provides real-time stock data visualization and AI-powered price predictions.

## Features

- **Real-time Stock Data**: Live stock prices and market data
- **Interactive Charts**: Historical price charts with multiple time periods
- **AI Predictions**: Machine learning-powered price predictions with confidence scores
- **Currency Support**: USD and INR currency conversion
- **Responsive Design**: Mobile-first design with dark/light theme support
- **Modern UI**: Built with Radix UI components and Tailwind CSS
- **Branded Experience**: Custom logo and favicon integration

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI primitives
- **Charts**: Recharts
- **Icons**: Lucide React
- **State Management**: React hooks

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on port 5000

### Installation

1. Clone the repository and navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Copy environment variables:
```bash
cp env.example .env.local
```

4. Update the environment variables in `.env.local`:
```bash
BACKEND_URL=http://localhost:5000
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### Development

Start the development server:
```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

Build the application for production:
```bash
npm run build
# or
yarn build
```

Start the production server:
```bash
npm start
# or
yarn start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── api/               # API routes
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Home page
│   ├── components/            # React components
│   │   ├── ui/               # UI components (Radix UI)
│   │   ├── StockDashboard.tsx
│   │   ├── StockSearch.tsx
│   │   ├── StockInfo.tsx
│   │   ├── StockChart.tsx
│   │   ├── StockPrediction.tsx
│   │   └── CurrencyToggle.tsx
│   ├── services/             # API services
│   │   └── stockService.ts
│   └── utils/                # Utility functions
│       └── currency.ts
├── public/                   # Static assets
│   ├── logo.png             # Main project logo
│   ├── logo.svg             # Vector logo
│   ├── bull-logo.svg        # Bull logo variant
│   └── favicon.ico          # Browser favicon
├── next.config.js           # Next.js configuration
├── tailwind.config.js       # Tailwind CSS configuration
├── tsconfig.json           # TypeScript configuration
└── package.json
```

## API Integration

The frontend communicates with the backend through Next.js API routes that proxy requests:

- `/api/backend/stock/*` - Stock data endpoints
- `/api/backend/predict/*` - Prediction endpoints
- `/api/backend/health` - Health check endpoint

## Key Components

### StockDashboard
Main dashboard component that orchestrates all other components and manages state.

### StockSearch
Stock search functionality with popular stocks and real-time search results.

### StockInfo
Displays detailed stock information including price, volume, market cap, etc.

### StockChart
Interactive price charts with multiple time periods (1W, 1M, 1Y).

### StockPrediction
AI-powered price predictions with confidence scores and technical indicators.

### CurrencyToggle
Currency conversion between USD and INR.

## Styling

The application uses:
- **Tailwind CSS** for utility-first styling
- **CSS Variables** for theming (light/dark mode)
- **Radix UI** for accessible component primitives
- **Custom CSS** for advanced styling needs

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_URL` | Backend API URL | `http://localhost:5000` |
| `NEXT_PUBLIC_APP_URL` | Frontend URL | `http://localhost:3000` |

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy automatically

### Other Platforms

Build the application and deploy the `out` directory to any static hosting service.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
