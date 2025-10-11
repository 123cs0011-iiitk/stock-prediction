"use client"

import { TrendingUp, TrendingDown, Brain, Target, Calendar, AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { PredictionResult } from '@/services/stockService';
import { formatPrice, Currency } from '@/utils/currency';

interface StockPredictionProps {
  prediction: PredictionResult | null;
  currentPrice?: number;
  loading: boolean;
  symbol: string;
  error: string;
  currency: Currency;
}

export function StockPrediction({ 
  prediction, 
  currentPrice, 
  loading, 
  symbol, 
  error,
  currency 
}: StockPredictionProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Price Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-8 w-24" />
          </div>
          <Skeleton className="h-4 w-full" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Price Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Price Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            Select a stock to see AI-powered price predictions
          </p>
        </CardContent>
      </Card>
    );
  }

  const isPositiveChange = prediction.prediction.trend === 'bullish';
  const change = (currentPrice && prediction.current_price) ? prediction.current_price - currentPrice : 0;
  const changePercent = (currentPrice && change !== 0) ? ((change / currentPrice) * 100) : 0;

  const getConfidenceColor = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return 'text-muted-foreground';
    if (confidence >= 70) return 'text-green-600';
    if (confidence >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadgeVariant = (confidence: number | undefined | null): "default" | "secondary" | "destructive" | "outline" => {
    if (!confidence || isNaN(confidence)) return 'outline';
    if (confidence >= 70) return 'default';
    if (confidence >= 50) return 'secondary';
    return 'destructive';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          AI Price Prediction for {symbol}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Prediction Alert */}
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            This prediction is based on historical data analysis using machine learning. 
            Market conditions can change rapidly and predictions may not reflect future performance.
          </AlertDescription>
        </Alert>

        {/* Main Prediction */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Prediction Trend</span>
            <div className="flex items-center gap-2">
              {isPositiveChange ? (
                <TrendingUp className="w-4 h-4 text-green-600" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-600" />
              )}
              <Badge variant={getConfidenceBadgeVariant(prediction.prediction.confidence)}>
                {prediction.prediction.confidence ? prediction.prediction.confidence.toFixed(1) : '0.0'}% confidence
              </Badge>
            </div>
          </div>
          
          <div className="text-2xl font-bold capitalize">
            {prediction.prediction.trend}
          </div>
          
          {currentPrice && (
            <div className={`text-sm font-medium ${
              isPositiveChange ? 'text-green-600' : 'text-red-600'
            }`}>
              Current Price: {formatPrice(prediction.current_price, currency)}
            </div>
          )}
        </div>

        {/* Confidence Indicator */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Prediction Confidence</span>
            <span className={`text-sm font-medium ${getConfidenceColor(prediction.prediction.confidence)}`}>
              {prediction.prediction.confidence ? prediction.prediction.confidence.toFixed(1) : '0.0'}%
            </span>
          </div>
          <Progress value={prediction.prediction.confidence || 0} className="h-2" />
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Support Level</span>
            </div>
            <div className="text-sm font-medium">
              {formatPrice(prediction.prediction.support_level, currency)}
            </div>
          </div>
          
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Resistance Level</span>
            </div>
            <div className="text-sm font-medium">
              {formatPrice(prediction.prediction.resistance_level, currency)}
            </div>
          </div>
        </div>

        {/* Next Day Range */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Next Day Range</span>
          </div>
          <div className="text-sm font-medium">
            {formatPrice(prediction.prediction.next_day_range.low, currency)} - {formatPrice(prediction.prediction.next_day_range.high, currency)}
          </div>
        </div>

        {/* Model Info */}
        <div className="pt-4 border-t space-y-2">
          <h4 className="font-medium text-sm">Model Information</h4>
          <div className="text-xs text-muted-foreground space-y-1">
            <p><strong>ML Models:</strong> {prediction.model_status.ml_models_trained ? 'Trained' : 'Not Trained'}</p>
            <p><strong>ARIMA Model:</strong> {prediction.model_status.arima_trained ? 'Trained' : 'Not Trained'}</p>
            <p><strong>SMA 20:</strong> {formatPrice(prediction.technical_indicators.sma_20, currency)}</p>
            <p><strong>SMA 50:</strong> {formatPrice(prediction.technical_indicators.sma_50, currency)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
