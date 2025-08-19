# Unicode Escaping Fix for Arabic Text in JSON Responses

## Problem Analysis

The issue is that FastAPI's JSONResponse is escaping Unicode characters by default, converting Arabic text to Unicode escape sequences like `\u062a\u0644\u0643` instead of keeping the actual UTF-8 encoded Arabic characters. This causes display issues in OpenWebUI.

## Root Cause

FastAPI's JSONResponse uses Python's `json.dumps()` function which by default has `ensure_ascii=True`, causing all non-ASCII characters to be escaped as Unicode sequences.

## Solution Approach

We need to modify the JSON response handling in `web_api.py` to ensure that Unicode characters are not escaped, allowing proper UTF-8 encoded Arabic text to be returned.

## Implementation Options

### Option 1: Custom JSON Encoder (Recommended)

Create a custom JSONResponse class that uses `ensure_ascii=False`:

```python
import json
from fastapi.responses import JSONResponse

class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
```

### Option 2: Manual JSON Serialization

Handle JSON serialization manually in the endpoint and return a Response with proper content:

```python
from fastapi import Response
import json

# In the endpoint
json_content = json.dumps(response_data, ensure_ascii=False)
return Response(content=json_content, media_type="application/json; charset=utf-8")
```

## Implementation Steps

### Step 1: Modify web_api.py

Update the `/invoke` endpoint to use proper UTF-8 JSON serialization without Unicode escaping.

### Step 2: Test the Fix

Verify that Arabic text is properly returned without Unicode escaping.

### Step 3: Verify with OpenWebUI

Test that OpenWebUI properly displays the Arabic text.

## Code Changes

In `web_api.py`, we need to modify the `invoke_endpoint` function:

```python
# Add this import at the top
import json
from fastapi.responses import Response

# Replace the invoke_endpoint function
@app.post("/invoke", response_model=ToolOutput)
async def invoke_endpoint(input: ToolInput):
    if PIPELINE is None:
        raise HTTPException(status_code=500, detail="Pipeline failed to initialize")
    try:
        result = PIPELINE.query(input.query, input.target_lang, input.return_original)
        response_data = {
            "response": result["original_response"],
            "translation": result.get("translation"),
            "source_language": result["source_language"]
        }
        # Add target_language only if target_lang was requested
        if input.target_lang is not None:
            response_data["target_language"] = input.target_lang
        
        # Serialize JSON with ensure_ascii=False to prevent Unicode escaping
        json_content = json.dumps(response_data, ensure_ascii=False)
        return Response(
            content=json_content,
            media_type="application/json; charset=utf-8"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Expected Benefits

1. Arabic text will be properly encoded as UTF-8 without Unicode escaping
2. OpenWebUI will be able to properly display Arabic text
3. No more Unicode escape sequences like `\u062a\u0644\u0643` in the response
4. Better compatibility with international text in general

## Testing

After implementing the fix, we should:

1. Test the API endpoint directly to verify Arabic text is properly encoded
2. Test with OpenWebUI to verify proper display
3. Verify that other languages still work correctly
4. Check that the response headers are still correct

## Rollback Plan

If issues arise with the new JSON serialization approach:

1. Revert to the original JSONResponse implementation
2. Add configuration option to control Unicode escaping behavior
3. Implement logging to track encoding issues