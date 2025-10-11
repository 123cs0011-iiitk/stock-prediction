/**
 * Frontend test snippet for live stock price integration
 * Run this in the browser console to test the live price functionality
 */

// Test function to verify live stock price API
async function testLiveStockPrice() {
    console.log('🧪 Testing Live Stock Price Integration');
    console.log('=====================================');
    
    const testSymbol = 'AAPL';
    const baseUrl = 'http://localhost:5000/api';
    
    try {
        console.log(`📊 Fetching live price for ${testSymbol}...`);
        
        const response = await fetch(`${baseUrl}/stock/price/${testSymbol}`, {
            headers: {
                'Content-Type': 'application/json',
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        console.log('✅ Live price data retrieved successfully:');
        console.log(`   Symbol: ${data.symbol}`);
        console.log(`   Name: ${data.name}`);
        console.log(`   Price: $${data.price}`);
        console.log(`   Change: $${data.change} (${data.changePercent}%)`);
        console.log(`   Volume: ${data.volume.toLocaleString()}`);
        console.log(`   High: $${data.high}`);
        console.log(`   Low: $${data.low}`);
        console.log(`   Open: $${data.open}`);
        console.log(`   Previous Close: $${data.previousClose}`);
        console.log(`   Market Cap: ${data.marketCap}`);
        console.log(`   Sector: ${data.sector}`);
        console.log(`   Industry: ${data.industry}`);
        console.log(`   Source: ${data.source}`);
        console.log(`   Timestamp: ${data.timestamp}`);
        
        return data;
        
    } catch (error) {
        console.error('❌ Failed to fetch live stock price:', error);
        
        if (error.message.includes('Failed to fetch')) {
            console.log('💡 Make sure the backend server is running:');
            console.log('   cd backend && python app.py');
        }
        
        throw error;
    }
}

// Test function using the stockService
async function testStockService() {
    console.log('\n🔧 Testing StockService Integration');
    console.log('===================================');
    
    try {
        // Import the stock service (this would work in the actual app)
        console.log('📝 Note: This test requires the React app to be running');
        console.log('   Run: npm run start (in frontend directory)');
        console.log('   Then test the search functionality in the UI');
        
        // Simulate what the stockService would do
        const testSymbol = 'MSFT';
        const response = await fetch(`http://localhost:5000/api/stock/price/${testSymbol}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log(`✅ StockService would return: ${data.symbol} at $${data.price}`);
        }
        
    } catch (error) {
        console.error('❌ StockService test failed:', error);
    }
}

// Helper function to test multiple symbols
async function testMultipleSymbols() {
    console.log('\n🔄 Testing Multiple Stock Symbols');
    console.log('==================================');
    
    const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];
    
    for (const symbol of symbols) {
        try {
            console.log(`\n📈 Testing ${symbol}...`);
            const response = await fetch(`http://localhost:5000/api/stock/price/${symbol}`);
            
            if (response.ok) {
                const data = await response.json();
                console.log(`   ✅ ${symbol}: $${data.price} (${data.changePercent}%)`);
            } else {
                console.log(`   ❌ ${symbol}: HTTP ${response.status}`);
            }
            
            // Wait between requests to respect rate limits
            await new Promise(resolve => setTimeout(resolve, 2000));
            
        } catch (error) {
            console.log(`   ❌ ${symbol}: ${error.message}`);
        }
    }
}

// Main test runner
async function runAllTests() {
    console.log('🚀 Live Stock Price Integration Tests');
    console.log('=====================================');
    
    try {
        await testLiveStockPrice();
        await testStockService();
        await testMultipleSymbols();
        
        console.log('\n🎉 All tests completed!');
        console.log('\n📝 Next steps:');
        console.log('1. Start the backend: cd backend && python app.py');
        console.log('2. Start the frontend: cd frontend && npm run start');
        console.log('3. Open http://localhost:3000 and search for any stock symbol');
        console.log('4. Verify that live prices are displayed');
        
    } catch (error) {
        console.error('\n❌ Test suite failed:', error);
    }
}

// Export functions for manual testing
window.testLiveStockPrice = testLiveStockPrice;
window.testStockService = testStockService;
window.testMultipleSymbols = testMultipleSymbols;
window.runAllTests = runAllTests;

console.log('🔧 Test functions loaded! Run one of these:');
console.log('   testLiveStockPrice() - Test single stock');
console.log('   testMultipleSymbols() - Test multiple stocks');
console.log('   runAllTests() - Run complete test suite');
