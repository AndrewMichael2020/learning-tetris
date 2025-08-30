#!/usr/bin/env python3
"""
Test WebSocket messages to see exactly what's being sent and received.
"""

import asyncio
import websockets
import json

async def test_play_once_detailed():
    uri = "ws://localhost:8000/ws/play-once?algo=cem"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            print("Listening for messages...")
            
            message_count = 0
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\nMessage {message_count}:")
                    print(f"  Raw keys: {list(data.keys())}")
                    
                    # Check specific fields
                    if 'score' in data:
                        print(f"  score: {data['score']} (type: {type(data['score'])})")
                    if 'lines' in data:
                        print(f"  lines: {data['lines']} (type: {type(data['lines'])})")  
                    if 'step' in data:
                        print(f"  step: {data['step']} (type: {type(data['step'])})")
                    if 'done' in data:
                        print(f"  done: {data['done']} (type: {type(data['done'])})")
                    if 'final' in data:
                        print(f"  final: {data['final']} (type: {type(data['final'])}) *** FINAL MESSAGE ***")
                        print(f"  Final data: score={data.get('score')}, lines={data.get('lines')}, steps={data.get('steps')}")
                        break
                        
                    if message_count > 30:  # Avoid infinite loops
                        print("Stopping after 30 messages")
                        break
                        
                except asyncio.TimeoutError:
                    print("No message received within 5 seconds, continuing...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_play_once_detailed())
