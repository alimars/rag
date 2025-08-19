#!/usr/bin/env python3
"""
Test script to verify that the Unicode escaping fix works correctly for Arabic text
"""
import requests
import json

def test_arabic_unicode_fix():
    """Test that Arabic text is properly encoded without Unicode escaping"""
    url = "http://localhost:8000/invoke"
    
    # Test case: Arabic query with return_original=True to get Arabic document content
    print("Testing Arabic text encoding...")
    arabic_query = "ما هي القضايا الرئيسية في هذا المستند؟"
    payload = {
        "query": arabic_query,
        "return_original": True
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Get raw response content to check encoding
            raw_content = response.content.decode('utf-8')
            print(f"Raw response length: {len(raw_content)} characters")
            
            # Check if we have Unicode escape sequences
            has_unicode_escapes = "\\u06" in raw_content  # Arabic Unicode range
            print(f"Contains Unicode escapes: {has_unicode_escapes}")
            
            # Parse JSON to get structured data
            result = response.json()
            print(f"Source language: {result.get('source_language')}")
            
            # Check response content
            response_text = result.get("response", "")
            print(f"Response length: {len(response_text)} characters")
            
            # Check if response contains Arabic characters directly (not escaped)
            # Arabic Unicode range: U+0600 to U+06FF
            has_arabic_chars = any('\u0600' <= char <= '\u06FF' for char in response_text)
            print(f"Contains direct Arabic characters: {has_arabic_chars}")
            
            # Show a sample of the response
            sample = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"Response sample: {sample}")
            
            if has_unicode_escapes:
                print("❌ FAIL: Response still contains Unicode escape sequences")
                return False
            elif has_arabic_chars:
                print("✅ PASS: Response contains direct Arabic characters without Unicode escaping")
                return True
            else:
                print("⚠️  WARNING: Response doesn't clearly show Arabic content")
                return False
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_english_still_works():
    """Test that English text still works correctly"""
    url = "http://localhost:8000/invoke"
    
    print("\nTesting English text encoding...")
    english_query = "What are the main issues in these documents?"
    payload = {
        "query": english_query,
        "return_original": True
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print(f"Response length: {len(response_text)} characters")
            print(f"Source language: {result.get('source_language')}")
            
            # Show a sample of the response
            sample = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"Response sample: {sample}")
            
            print("✅ PASS: English text still works correctly")
            return True
        else:
            print(f"❌ FAIL: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Testing Unicode escaping fix for Arabic text...\n")
    
    arabic_success = test_arabic_unicode_fix()
    english_success = test_english_still_works()
    
    print("\n" + "="*50)
    if arabic_success and english_success:
        print("🎉 ALL TESTS PASSED: Unicode escaping fix is working correctly!")
    else:
        print("❌ SOME TESTS FAILED: Please check the implementation")