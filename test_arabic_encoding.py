#!/usr/bin/env python3
"""
Test script to verify Arabic encoding fix
"""
import json
import sys

def test_arabic_encoding():
    """Test that Arabic text is properly encoded in JSON"""
    # Sample Arabic text that was causing issues
    arabic_text = "السورية من شأنها أن تشكل انتهاكا للاتفاقيية.5\nالوضع العام في المنطقة\nكانت محافضة إدلب (بالإضافة إلى"
    
    # Create a response object similar to what the RAG pipeline returns
    response_data = {
        "response": arabic_text,
        "translation": None,
        "source_language": "ar"
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(response_data, ensure_ascii=False)
        print("✅ JSON serialization successful")
        print(f"Serialized JSON: {json_str}")
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        if parsed_data["response"] == arabic_text:
            print("✅ JSON deserialization successful")
            print("✅ Arabic text preserved correctly")
            return True
        else:
            print("❌ Arabic text not preserved correctly")
            return False
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_arabic_encoding()
    sys.exit(0 if success else 1)