#!/usr/bin/env python3
"""
Test script to verify the Arabic query translation fix
"""
import requests
import json

def test_arabic_query():
    """Test Arabic query with Arabic documents"""
    url = "http://localhost:8000/query"
    
    # Test case 1: Arabic query with no target language (should return Arabic response)
    arabic_query = "ما هي القضايا الرئيسية في هذا المستند؟"
    payload = {
        "text": arabic_query
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Test 1 - Arabic query, no target language:")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Error in Test 1: {e}")
        print()
    
    # Test case 2: Arabic query with English target language
    payload_with_target = {
        "text": arabic_query,
        "target_lang": "en"
    }
    
    try:
        response = requests.post(url, json=payload_with_target)
        print("Test 2 - Arabic query, English target language:")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Error in Test 2: {e}")
        print()
    
    # Test case 3: Arabic query with return_original=True
    print("Test 3 - Arabic query, return original content:")
    payload_with_original = {
        "text": arabic_query,
        "return_original": True
    }
    
    try:
        response = requests.post(url, json=payload_with_original)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            # Check that translation is None or not present
            if result.get("translation") is None:
                print("✅ PASS: translation field is correctly None")
            else:
                print(f"❌ FAIL: translation field should be None, got '{result.get('translation')}'")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Error in Test 3: {e}")
        print()

if __name__ == "__main__":
    test_arabic_query()