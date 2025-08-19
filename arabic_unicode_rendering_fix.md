# Arabic Unicode Rendering Fix Plan

## Problem Analysis

The issue is not with language detection as initially thought, but with how Arabic text is being rendered in OpenWebUI. The RAG tool is correctly:

1. Loading Arabic documents
2. Detecting the language as Arabic
3. Returning Arabic content in the response

However, the Arabic text is being returned as Unicode escape sequences (like `\u062a\u0644\u0643`) rather than properly decoded UTF-8 text, and OpenWebUI is not properly rendering these sequences.

## Root Cause

1. The JSON serialization in the web API may be escaping Unicode characters unnecessarily
2. OpenWebUI may not be properly decoding Unicode escape sequences
3. There may be encoding issues in the communication between the RAG tool and OpenWebUI

## Current Status Check

Looking at the web_api.py file, I can see that it already includes:
```python
return JSONResponse(content=response_data, headers={"Content-Type": "application/json; charset=utf-8"})
```

This should ensure proper UTF-8 encoding, but there might be an issue with how ensure_ascii is being handled in the JSON serialization.

## Proposed Solution

### 1. Fix JSON Response Serialization

Modify the web API to ensure Unicode characters are not unnecessarily escaped:

In `web_api.py`, explicitly set `ensure_ascii=False` in the JSONResponse:

```python
# In the /invoke endpoint
return JSONResponse(
    content=response_data, 
    headers={"Content-Type": "application/json; charset=utf-8"},
    ensure_ascii=False
)
```

Actually, JSONResponse doesn't have an ensure_ascii parameter. We need to handle this differently.

The issue is likely in how FastAPI/Starlette serializes the JSON. We should ensure that the content is properly UTF-8 encoded without Unicode escaping.

### 2. Alternative Approach - Custom JSON Encoder

If the above doesn't work, we can implement a custom JSON encoder that ensures proper handling of Arabic text.

### 3. OpenWebUI Configuration

The issue might also be in how OpenWebUI is configured to handle UTF-8 responses from custom tools.

## Implementation Steps

### Step 1: Modify web_api.py

Update the JSON response handling to ensure proper UTF-8 encoding without Unicode escaping.

### Step 2: Test with Direct API Calls

Verify that the API returns properly formatted Arabic text when called directly (which you've already confirmed works with Postman).

### Step 3: Check OpenWebUI Integration

Verify if there are any OpenWebUI-specific settings that need to be adjusted for proper UTF-8 handling.

## Expected Benefits

1. Arabic text will be properly displayed in OpenWebUI
2. No more Unicode escape sequences in the response
3. Proper rendering of Arabic documents

## Rollback Plan

If issues arise with the new encoding approach:
1. Revert to the original JSON response handling
2. Add configuration option to control encoding behavior
3. Implement logging to track encoding issues