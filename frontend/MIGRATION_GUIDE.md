# Migration Guide: Vite React to Next.js

This guide documents the migration from the original Vite-based React frontend to the new Next.js frontend.

## Overview

The migration transforms the stock prediction dashboard from a Vite-based React SPA to a modern Next.js application with App Router, improved performance, and better developer experience.

## Key Changes

### 1. Framework Migration
- **From**: Vite + React SPA
- **To**: Next.js 14 with App Router
- **Benefits**: Server-side rendering, automatic code splitting, improved SEO

### 2. Project Structure
```
# Before (Vite)
frontend/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/
│   ├── services/
│   └── utils/
├── index.html
├── vite.config.ts
└── package.json

# After (Next.js)
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── globals.css
│   │   └── api/
│   ├── components/
│   ├── services/
│   └── utils/
├── next.config.js
├── tailwind.config.js
└── package.json
```

### 3. Component Changes

#### App.tsx → StockDashboard.tsx
- Renamed for clarity
- Added "use client" directive for client-side functionality
- Updated import paths to use Next.js aliases

#### Import Path Updates
```typescript
// Before
import { stockService } from '../services/stockService';
import { formatPrice } from '../utils/currency';

// After
import { stockService } from '@/services/stockService';
import { formatPrice } from '@/utils/currency';
```

### 4. API Integration

#### Before (Direct Backend Calls)
```typescript
const BASE_URL = 'http://localhost:5000/api';
```

#### After (Next.js API Routes)
```typescript
const BASE_URL = '/api/backend';
// Proxied through Next.js API routes
```

### 5. Configuration Files

#### Vite Config → Next.js Config
```javascript
// vite.config.ts (removed)
export default defineConfig({
  plugins: [react()],
  resolve: { /* ... */ },
  server: { port: 3000 }
});

// next.config.js (new)
const nextConfig = {
  experimental: { appDir: true },
  async rewrites() {
    return [{
      source: '/api/backend/:path*',
      destination: 'http://localhost:5000/api/:path*'
    }]
  }
}
```

### 6. Styling Updates

#### Global CSS
- Moved from `src/index.css` to `src/app/globals.css`
- Updated Tailwind imports for Next.js
- Added CSS variables for theming

#### Component Styling
- All components remain identical in styling
- Added "use client" directives where needed
- Maintained responsive design and dark mode support

### 7. Dependencies

#### Added
- `next@^14.0.0` - Next.js framework
- `tailwindcss-animate@^1.0.7` - Animation utilities
- `eslint-config-next@^14.0.0` - Next.js ESLint config

#### Removed
- `vite@6.3.5` - Vite bundler
- `@vitejs/plugin-react-swc@^3.10.2` - Vite React plugin

#### Updated
- All other dependencies maintained compatibility

### 8. Development Workflow

#### Before
```bash
npm run dev    # Vite dev server
npm run build  # Vite build
npm run preview # Vite preview
```

#### After
```bash
npm run dev    # Next.js dev server
npm run build  # Next.js build
npm run start  # Next.js production server
npm run lint   # ESLint
```

### 9. Environment Variables

#### Before
```bash
# No environment variables needed
```

#### After
```bash
# .env.local
BACKEND_URL=http://localhost:5000
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

## Migration Benefits

### 1. Performance
- **Automatic Code Splitting**: Pages load faster with smaller bundles
- **Image Optimization**: Built-in image optimization
- **Font Optimization**: Automatic font loading optimization

### 2. Developer Experience
- **Hot Reloading**: Faster development with Next.js dev server
- **TypeScript**: Better TypeScript integration
- **ESLint**: Built-in linting with Next.js rules

### 3. Production Ready
- **Server-Side Rendering**: Better SEO and initial load performance
- **Static Generation**: Pre-rendered pages for better performance
- **API Routes**: Built-in API functionality

### 4. Deployment
- **Vercel Integration**: One-click deployment to Vercel
- **Static Export**: Can be deployed to any static hosting service
- **Docker Support**: Easy containerization

## Testing the Migration

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Setup Environment
```bash
cp env.example .env.local
# Update BACKEND_URL if needed
```

### 3. Start Development Server
```bash
npm run dev
```

### 4. Verify Features
- ✅ Stock search functionality
- ✅ Stock information display
- ✅ Interactive charts
- ✅ AI predictions
- ✅ Currency conversion
- ✅ Responsive design
- ✅ Dark/light theme

## Rollback Plan

If issues arise, you can easily rollback by:

1. Keep the original `frontend/` directory intact
2. Switch back to the Vite version by running:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Support

For issues with the migration:
1. Check the Next.js documentation
2. Verify environment variables
3. Ensure backend is running on correct port
4. Check browser console for errors

## Conclusion

The migration to Next.js provides significant improvements in performance, developer experience, and deployment options while maintaining all existing functionality. The application is now more scalable and production-ready.
