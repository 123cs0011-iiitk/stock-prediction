import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { BarChart3 } from 'lucide-react';
import { PricePoint } from '../services/stockService';
import { convertPrice, formatPrice, Currency } from '../utils/currency';

interface StockChartProps {
  data: PricePoint[];
  symbol: string;
  onPeriodChange: (period: 'week' | 'month' | 'year') => void;
  currentPeriod: 'week' | 'month' | 'year';
  loading: boolean;
  error: string;
  currency: Currency;
}

export function StockChart({ 
  data, 
  symbol, 
  onPeriodChange, 
  currentPeriod, 
  loading, 
  error,
  currency 
}: StockChartProps) {
  const periods = [
    { value: 'week' as const, label: '1W' },
    { value: 'month' as const, label: '1M' },
    { value: 'year' as const, label: '1Y' }
  ];

  // Convert price data to selected currency
  const convertedData = data.map(point => ({
    ...point,
    price: convertPrice(point.price, 'USD', currency)
  })).filter(point => point.price != null && !isNaN(point.price));

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium mb-1">{new Date(label).toLocaleDateString()}</p>
          <p className="text-sm text-primary">
            Price: {formatPrice(payload[0].value, currency)}
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Price Chart - {symbol}
            </CardTitle>
            <div className="flex gap-1">
              {periods.map(period => (
                <Skeleton key={period.value} className="h-8 w-12" />
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Price Chart - {symbol}
            </CardTitle>
            <div className="flex gap-1">
              {periods.map(period => (
                <Button
                  key={period.value}
                  variant={currentPeriod === period.value ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onPeriodChange(period.value)}
                >
                  {period.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (convertedData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Price Chart - {symbol}
            </CardTitle>
            <div className="flex gap-1">
              {periods.map(period => (
                <Button
                  key={period.value}
                  variant={currentPeriod === period.value ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onPeriodChange(period.value)}
                >
                  {period.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            No chart data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const minPrice = Math.min(...convertedData.map(d => d.price));
  const maxPrice = Math.max(...convertedData.map(d => d.price));
  const padding = (maxPrice - minPrice) * 0.1;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Price Chart - {symbol}
          </CardTitle>
          <div className="flex gap-1">
            {periods.map(period => (
              <Button
                key={period.value}
                variant={currentPeriod === period.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => onPeriodChange(period.value)}
              >
                {period.label}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={convertedData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
                className="text-xs"
              />
              <YAxis 
                domain={[minPrice - padding, maxPrice + padding]}
                tickFormatter={(value) => formatPrice(value, currency)}
                className="text-xs"
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: 'hsl(var(--primary))', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}