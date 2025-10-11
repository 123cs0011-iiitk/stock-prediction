import { useState, useEffect } from "react";
import { BarChart3, AlertCircle } from "lucide-react";
import { StockSearch } from "./components/StockSearch";
import { StockInfo } from "./components/StockInfo";
import { StockChart } from "./components/StockChart";
import { StockPrediction } from "./components/StockPrediction";
import { CurrencyToggle } from "./components/CurrencyToggle";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "./components/ui/card";
import { Alert, AlertDescription } from "./components/ui/alert";
import {
  stockService,
  StockData,
  PricePoint,
  PredictionResult,
} from "./services/stockService";
import { Currency } from "./utils/currency";

export default function App() {
  const [selectedSymbol, setSelectedSymbol] =
    useState<string>("");
  const [stockData, setStockData] = useState<StockData | null>(
    null,
  );
  const [chartData, setChartData] = useState<PricePoint[]>([]);
  const [prediction, setPrediction] =
    useState<PredictionResult | null>(null);
  const [chartPeriod, setChartPeriod] = useState<
    "week" | "month" | "year"
  >("month");
  const [currency, setCurrency] = useState<Currency>("USD");
  const [loading, setLoading] = useState({
    stock: false,
    chart: false,
    prediction: false,
  });
  const [errors, setErrors] = useState({
    stock: "",
    chart: "",
    prediction: "",
  });

  // Load stock data when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      loadStockData(selectedSymbol);
      loadChartData(selectedSymbol, chartPeriod);
      loadPrediction(selectedSymbol);
    }
  }, [selectedSymbol]);

  // Reload chart data when period changes
  useEffect(() => {
    if (selectedSymbol) {
      loadChartData(selectedSymbol, chartPeriod);
    }
  }, [chartPeriod, selectedSymbol]);

  const loadStockData = async (symbol: string) => {
    setLoading((prev) => ({ ...prev, stock: true }));
    setErrors((prev) => ({ ...prev, stock: "" }));
    try {
      // Try to get live price first, fall back to mock data if needed
      let data: StockData;
      try {
        data = await stockService.getLiveStockPrice(symbol);
      } catch (liveError) {
        console.log(`Live price not available for ${symbol}, using mock data`);
        data = await stockService.getStockData(symbol);
      }
      setStockData(data);
    } catch (error) {
      console.error("Failed to load stock data:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to load stock data";
      setErrors((prev) => ({ ...prev, stock: errorMessage }));
      setStockData(null);
    } finally {
      setLoading((prev) => ({ ...prev, stock: false }));
    }
  };

  const loadChartData = async (
    symbol: string,
    period: "week" | "month" | "year",
  ) => {
    setLoading((prev) => ({ ...prev, chart: true }));
    setErrors((prev) => ({ ...prev, chart: "" }));
    try {
      const data = await stockService.getHistoricalData(
        symbol,
        period,
      );
      setChartData(data);
    } catch (error) {
      console.error("Failed to load chart data:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to load chart data";
      setErrors((prev) => ({ ...prev, chart: errorMessage }));
      setChartData([]);
    } finally {
      setLoading((prev) => ({ ...prev, chart: false }));
    }
  };

  const loadPrediction = async (symbol: string) => {
    setLoading((prev) => ({ ...prev, prediction: true }));
    setErrors((prev) => ({ ...prev, prediction: "" }));
    try {
      const data = await stockService.getPrediction(symbol);
      setPrediction(data);
    } catch (error) {
      console.error("Failed to load prediction:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to generate prediction";
      setErrors((prev) => ({
        ...prev,
        prediction: errorMessage,
      }));
      setPrediction(null);
    } finally {
      setLoading((prev) => ({ ...prev, prediction: false }));
    }
  };

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  const handlePeriodChange = (
    period: "week" | "month" | "year",
  ) => {
    setChartPeriod(period);
  };

  const handleCurrencyChange = (newCurrency: Currency) => {
    setCurrency(newCurrency);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-4 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2 mb-2">
            <BarChart3 className="w-8 h-8 text-primary" />
            <h1>Stock Prediction Dashboard</h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Analyze real-time stock data and get AI-powered
            price predictions using machine learning algorithms.
            Always conduct your own research before making
            investment decisions.
          </p>

          {/* Currency Toggle */}
          <div className="flex justify-center mt-4">
            <CurrencyToggle
              currency={currency}
              onCurrencyChange={handleCurrencyChange}
            />
          </div>
        </div>

        {/* Warning Banner */}
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Investment Warning:</strong> Stock market
            predictions are inherently uncertain. This tool
            provides statistical analysis for educational
            purposes only. Past performance does not guarantee
            future results. Always consult with financial
            advisors and do thorough research before making
            investment decisions.
          </AlertDescription>
        </Alert>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Sidebar */}
          <div className="space-y-6">
            <StockSearch
              onStockSelect={handleStockSelect}
              selectedSymbol={selectedSymbol}
            />
            <StockInfo
              data={stockData}
              loading={loading.stock}
              error={errors.stock}
              currency={currency}
            />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <StockChart
              data={chartData}
              symbol={selectedSymbol || "Select a Stock"}
              onPeriodChange={handlePeriodChange}
              currentPeriod={chartPeriod}
              loading={loading.chart}
              error={errors.chart}
              currency={currency}
            />

            <StockPrediction
              prediction={prediction}
              currentPrice={stockData?.price}
              loading={loading.prediction}
              symbol={selectedSymbol}
              error={errors.prediction}
              currency={currency}
            />
          </div>
        </div>

        {/* Footer */}
        <Card>
          <CardHeader>
            <CardTitle>About This Tool</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              This stock prediction dashboard uses advanced machine learning algorithms
              to analyze historical price patterns and predict future price movements.
              The system combines Linear Regression, Random Forest Regressor, and ARIMA
              time series models to provide comprehensive predictions with confidence scores.
            </p>
            <div className="text-xs text-muted-foreground space-y-1">
              <p>
                <strong>Backend:</strong> Flask API with real-time ML processing
                and comprehensive error handling
              </p>
              <p>
                <strong>ML Algorithms:</strong> 
                <br/>• Linear Regression (30% weight)
                <br/>• Random Forest Regressor (70% weight) 
                <br/>• ARIMA Time Series Model
                <br/>• Ensemble Prediction Method
              </p>
              <p>
                <strong>Technical Indicators:</strong> RSI, SMA, Bollinger Bands,
                momentum, volatility, and price-volume correlation
              </p>
              <p>
                <strong>Data Processing:</strong> Real-time feature engineering
                with 20+ technical indicators from historical data
              </p>
              <p>
                <strong>Caching:</strong> Smart caching for performance - 
                stock data (5 min), predictions cached per symbol
              </p>
              <p>
                <strong>Currency Support:</strong> Real-time conversion between 
                USD and INR with live formatting
              </p>
              <p>
                <strong>Data Source:</strong> Live stock data via Alpha Vantage API
                with comprehensive error handling and rate limiting
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}