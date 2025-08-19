# Arabic Language Detection Fix Plan

## Problem Analysis

The current implementation uses the `langid` library for language detection, which may not be accurately detecting Arabic text. This leads to translation issues where Arabic queries are not properly identified as Arabic, causing incorrect translation behavior.

## Root Cause

1. The `langid` library may have limitations in accurately detecting Arabic text, especially with shorter texts or mixed content
2. No fallback mechanism exists when the primary detection method fails
3. The system relies solely on `langid` without any additional validation

## Proposed Solution

### 1. Add Alternative Language Detection Library

Add the `langdetect` library as an alternative detection method:
- More accurate for many languages including Arabic
- Provides confidence scores
- Can be used as a fallback when `langid` fails

### 2. Implement Hybrid Detection Approach

Create a more robust detection system that:
1. Uses multiple detection libraries
2. Implements a voting mechanism or confidence-based selection
3. Adds specific handling for known issues with Arabic text

### 3. Add Specific Arabic Text Validation

Implement additional checks for Arabic text:
- Unicode range validation for Arabic characters
- Length-based confidence adjustment
- Special handling for mixed-language content

## Implementation Steps

### Step 1: Update Requirements

Add `langdetect` to the requirements.txt file:
```
langdetect>=1.0.9
```

### Step 2: Modify Translation System

Update the `OfflineTranslationSystem` class in `translation.py`:

1. Import the new detection library:
```python
import langid
from langdetect import detect, detect_langs
import re
```

2. Enhance the `detect_language` method:
```python
def detect_language(self, text):
    # Primary detection with langid
    langid_result, langid_confidence = self.detector.classify(text)
    
    # Secondary detection with langdetect
    try:
        langdetect_results = detect_langs(text)
        langdetect_result = langdetect_results[0].lang
        langdetect_confidence = langdetect_results[0].prob
    except:
        langdetect_result = None
        langdetect_confidence = 0.0
    
    # Check for Arabic characters specifically
    arabic_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    has_arabic_chars = bool(arabic_pattern.search(text))
    
    # Special handling for Arabic text
    if has_arabic_chars and len(text.strip()) > 10:
        # If we have Arabic characters and text is reasonably long, 
        # prioritize Arabic detection
        if langid_result == 'ar' or langdetect_result == 'ar':
            return 'ar'
        # If neither detected Arabic but we have Arabic characters,
        # use a more lenient approach
        if has_arabic_chars:
            return 'ar'
    
    # Use confidence-based selection
    if langid_confidence > 0.9:
        return langid_result
    elif langdetect_confidence > 0.9:
        return langdetect_result
    elif langid_confidence > langdetect_confidence:
        return langid_result
    elif langdetect_confidence > 0:
        return langdetect_result
    else:
        return langid_result  # fallback to langid
```

### Step 3: Update Dockerfile

Add the necessary system dependencies for `langdetect` if needed.

### Step 4: Testing

Create comprehensive tests to verify the improved detection works correctly:
- Test with pure Arabic text
- Test with mixed Arabic/English text
- Test with short and long Arabic texts
- Verify that translation behavior is now correct

## Expected Benefits

1. More accurate Arabic language detection
2. Better handling of mixed-language content
3. Improved translation quality for Arabic queries
4. More robust language detection overall

## Rollback Plan

If issues arise with the new detection system:
1. Revert to the original `langid`-only approach
2. Add configuration option to select detection method
3. Implement logging to track detection accuracy