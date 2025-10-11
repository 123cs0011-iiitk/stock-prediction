"use client"

import Image from "next/image";

interface LogoProps {
  width?: number;
  height?: number;
  className?: string;
  showText?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export function Logo({ 
  width, 
  height, 
  className = "", 
  showText = false,
  size = 'md'
}: LogoProps) {
  const sizeMap = {
    sm: { width: 24, height: 24 },
    md: { width: 48, height: 48 },
    lg: { width: 64, height: 64 }
  };

  const dimensions = width && height ? { width, height } : sizeMap[size];

  if (showText) {
    return (
      <div className={`flex items-center gap-3 ${className}`}>
        <Image
          src="/logo.png"
          alt="Stock Price Insight Arena Logo"
          width={dimensions.width}
          height={dimensions.height}
          className="rounded-lg"
        />
        <div className="flex flex-col">
          <span className="text-lg font-bold">Stock Price Insight Arena</span>
          <span className="text-xs text-muted-foreground">AI-Powered Analysis</span>
        </div>
      </div>
    );
  }

  return (
    <Image
      src="/logo.png"
      alt="Stock Price Insight Arena Logo"
      width={dimensions.width}
      height={dimensions.height}
      className={`rounded-lg ${className}`}
    />
  );
}
