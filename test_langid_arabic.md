# Test Script for Arabic Language Detection

This script tests how the langid library performs with Arabic text detection.

```python
#!/usr/bin/env python3
"""
Test script to verify Arabic language detection with langid
"""
import langid

def test_arabic_detection():
    """Test Arabic language detection"""
    # Sample Arabic text
    arabic_text = "السورية من شأنها أن تشكل انتهاكا للاتفاقيية.5\nالوضع العام في المنطقة\nكانت محافضة إدلب (بالإضافة إلى"
    
    # Test language detection
    detected_lang, confidence = langid.classify(arabic_text)
    
    print(f"Input text: {arabic_text}")
    print(f"Detected language: {detected_lang}")
    print(f"Confidence: {confidence}")
    
    # Test with longer Arabic text
    longer_arabic_text = """
    اللغة العربية هي لغة闪ل من اللغات السامية، وتُعتبر من أكثر اللغات انتشاراً في العالم، 
    فهي اللغة الرسمية لعدد كبير من الدول، كما أنها لغة القرآن الكريم، مما جعلها لغة مهمة 
    في العالم الإسلامي. تتميز اللغة العربية بتمييزها بين الألفاظ والقواعد النحوية والصرفية 
    المعقدة، مما يجعلها لغة غنية ودقيقة في التعبير.
    """
    
    detected_lang2, confidence2 = langid.classify(longer_arabic_text)
    
    print(f"\nLonger Arabic text test:")
    print(f"Detected language: {detected_lang2}")
    print(f"Confidence: {confidence2}")
    
    # Test with mixed text
    mixed_text = "This is English text مع بعض النصوص العربية"
    detected_lang3, confidence3 = langid.classify(mixed_text)
    
    print(f"\nMixed text test:")
    print(f"Input text: {mixed_text}")
    print(f"Detected language: {detected_lang3}")
    print(f"Confidence: {confidence3}")

if __name__ == "__main__":
    test_arabic_detection()
```

To run this test, save it as `test_langid_arabic.py` and execute it in the environment where your RAG tool is running.