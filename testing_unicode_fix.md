# Testing the Unicode Escaping Fix for Arabic Text

## Overview
This document provides instructions for testing the Unicode escaping fix that was implemented to resolve Arabic text display issues in OpenWebUI.

## Prerequisites
- The RAG API service with the applied fix
- Python 3.x with requests library installed
- Access to OpenWebUI
- Arabic documents in the RAG tool's documents directory

## Testing Steps

### 1. Restart the RAG API Service
After applying the fix, you need to restart the RAG API service to apply the changes:

```bash
# If running with Docker
docker-compose restart rag-api

# Or if running directly
cd advanced-rag-offline
./start_rag.sh
```

### 2. Run the Test Script
Execute the test script to verify that the fix works correctly:

```bash
python test_arabic_unicode_fix.py
```

The script will:
- Send an Arabic query to the RAG API
- Check if the response contains direct Arabic characters (not Unicode escape sequences)
- Verify that English text still works correctly
- Report the results

### 3. Expected Results
A successful test should show:
- ✅ PASS: Response contains direct Arabic characters without Unicode escaping
- ✅ PASS: English text still works correctly
- 🎉 ALL TESTS PASSED: Unicode escaping fix is working correctly!

### 4. Manual API Testing
You can also test manually using curl:

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هي القضايا الرئيسية في هذا المستند؟", "return_original": true}'
```

Check the response to ensure it contains direct Arabic characters rather than Unicode escape sequences like `\u062a\u0644\u0643`.

### 5. Test with OpenWebUI
Finally, test with OpenWebUI to verify that Arabic text is properly displayed:

1. Open OpenWebUI
2. Select the RAG tool
3. Enter an Arabic query
4. Verify that the response shows proper Arabic text rather than Unicode escape sequences

## Troubleshooting

### If Tests Fail
1. Check that the RAG API service was restarted after applying the fix
2. Verify that the web_api.py file was modified correctly
3. Check the service logs for any errors:
   ```bash
   docker-compose logs rag-api
   ```

### If OpenWebUI Still Shows Issues
1. Clear the browser cache
2. Check browser developer tools for any JavaScript errors
3. Verify that the API response is correct using browser network tools

## Rollback Plan
If issues arise with the fix:

1. Revert the changes to web_api.py
2. Restart the RAG API service
3. The system will return to the previous behavior with Unicode escaping

## Additional Notes
- The fix should not affect other languages or functionality
- Performance should remain the same as before
- The fix only affects how Unicode characters are serialized in JSON responses