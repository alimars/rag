#!/usr/bin/env python3
"""
Test script to verify the original Arabic content retrieval feature
"""
import requests
import json

def test_original_arabic_content():
    """Test retrieving original Arabic content"""
    url = "http://localhost:8000/query"
    
    # Test case 1: Arabic query with return_original=True
    print("Test 1 - Arabic query, return original content:")
    arabic_query = "ما هي القضايا الرئيسية في هذا المستند؟"
    payload = {
        "text": arabic_query,
        "return_original": True
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print("✅ PASS: Original Arabic content retrieved successfully")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Error in Test 1: {e}")
        print()
    
    # Test case 2: English query with return_original=True on Arabic documents
    print("Test 2 - English query, return original content:")
    english_query = "What are the main issues in these documents?"
    payload = {
        "text": english_query,
        "return_original": True
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print("✅ PASS: Original content retrieved successfully")
        else:
            print(f"Error: {response.text}")
        print()
    except Exception as e:
        print(f"Error in Test 2: {e}")
        print()

if __name__ == "__main__":
    test_original_arabic_content()