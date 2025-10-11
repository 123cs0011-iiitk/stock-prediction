// Exchange rate USD to INR (approximate rate for demo purposes)
// In a production app, this would be fetched from a real exchange rate API
const USD_TO_INR_RATE = 83.5;

export type Currency = 'USD' | 'INR';

export function convertPrice(price: number | undefined | null, fromCurrency: Currency = 'USD', toCurrency: Currency = 'USD'): number {
  // Handle null/undefined prices
  if (price == null || isNaN(price)) {
    return 0;
  }
  
  if (fromCurrency === toCurrency) {
    return price;
  }
  
  if (fromCurrency === 'USD' && toCurrency === 'INR') {
    return price * USD_TO_INR_RATE;
  }
  
  if (fromCurrency === 'INR' && toCurrency === 'USD') {
    return price / USD_TO_INR_RATE;
  }
  
  return price;
}

export function formatPrice(price: number | undefined | null, currency: Currency = 'USD'): string {
  // Handle null/undefined prices
  if (price == null || isNaN(price)) {
    return currency === 'INR' ? '₹0.00' : '$0.00';
  }
  
  const converted = convertPrice(price, 'USD', currency);
  
  if (currency === 'INR') {
    return `₹${converted.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
  }
  
  return `${converted.toFixed(2)}`;
}

export function getCurrencySymbol(currency: Currency): string {
  return currency === 'USD' ? '$' : '₹';
}
