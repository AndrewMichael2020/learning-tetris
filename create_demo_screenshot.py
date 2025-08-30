#!/usr/bin/env python3
"""
Simple script to create a demo screenshot showing the app is working.
"""
import asyncio
import httpx
import json
from datetime import datetime

async def test_app():
    """Test the running app and display results."""
    print("🎮 AI-Powered Tetris Learning App - Demo Test Results")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            print("🔍 Testing health endpoint...")
            health_response = await client.get("http://localhost:8000/api/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health check passed: {health_data}")
            else:
                print(f"❌ Health check failed: {health_response.status_code}")
                return
            
            # Test play endpoint
            print("\n🤖 Testing AI agent play...")
            play_request = {
                "algorithm": "cem",
                "episodes": 1,
                "seed": 42
            }
            
            play_response = await client.post(
                "http://localhost:8000/api/play",
                json=play_request,
                timeout=30.0
            )
            
            if play_response.status_code == 200:
                play_data = play_response.json()
                print(f"✅ AI Play test successful!")
                print(f"   📊 Episodes completed: {len(play_data.get('results', []))}")
                if play_data.get('results'):
                    result = play_data['results'][0]
                    print(f"   🎯 Score achieved: {result.get('score', 'N/A')}")
                    print(f"   📏 Lines cleared: {result.get('lines_cleared', 'N/A')}")
                    print(f"   ⏱️  Steps taken: {result.get('steps', 'N/A')}")
            else:
                print(f"❌ AI Play test failed: {play_response.status_code}")
                print(f"   Error: {play_response.text}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure the server is running on http://localhost:8000")
    
    print("\n🌐 App Interface Available at: http://localhost:8000")
    print(f"🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_app())
