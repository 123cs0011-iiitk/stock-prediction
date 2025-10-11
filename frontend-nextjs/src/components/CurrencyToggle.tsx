"use client"

import { DollarSign, IndianRupee } from 'lucide-react';
import { Button } from './ui/button';

interface CurrencyToggleProps {
  currency: 'USD' | 'INR';
  onCurrencyChange: (currency: 'USD' | 'INR') => void;
}

export function CurrencyToggle({ currency, onCurrencyChange }: CurrencyToggleProps) {
  return (
    <div className="flex items-center gap-1 border rounded-lg p-1">
      <Button
        variant={currency === 'USD' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onCurrencyChange('USD')}
        className="h-8 px-3 gap-1"
      >
        <DollarSign className="w-4 h-4" />
        USD
      </Button>
      <Button
        variant={currency === 'INR' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => onCurrencyChange('INR')}
        className="h-8 px-3 gap-1"
      >
        <IndianRupee className="w-4 h-4" />
        INR
      </Button>
    </div>
  );
}
