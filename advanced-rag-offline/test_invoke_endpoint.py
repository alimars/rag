#!/usr/bin/env python3
"""
Test script for the /invoke endpoint with return_original=True
This script tests that the endpoint returns the correct JSON format.
"""

import requests
import json

def test_invoke_endpoint():
    """Test the /invoke endpoint with return_original=True"""
    
    # Endpoint URL - assuming the server is running locally on port 8000
    url = "http://localhost:8000/invoke"
    
    # Test query - using a simple query that should return results
    test_query = "What is the organization for the prohibition of chemical weapons?"
    
    # Request payload with return_original=True
    payload = {
        "query": test_query,
        "return_original": True
    }
    
    try:
        print(f"Making request to {url}...")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make the POST request
        response = requests.post(url, json=payload)
        
        # Print the response
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"ERROR: Request failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
        # Parse the JSON response
        try:
            # The response is a JSON string, so we need to parse it
            response_data = json.loads(response.text)
            print(f"\nResponse Data:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response: {e}")
            print(f"Response content: {response.text}")
            return False
            
        # Verify the response format
        print("\nVerifying response format...")
        
        # Check that response is a dictionary
        if not isinstance(response_data, dict):
            print("ERROR: Response is not a dictionary")
            return False
            
        # Check that 'ids' array is present
        if 'ids' not in response_data:
            print("ERROR: 'ids' field is missing from response")
            return False
            
        if not isinstance(response_data['ids'], list):
            print("ERROR: 'ids' field is not an array")
            return False
            
        print(f"✓ Found 'ids' array with {len(response_data['ids'])} items")
        
        # Check that 'documents' array is present
        if 'documents' not in response_data:
            print("ERROR: 'documents' field is missing from response")
            return False
            
        if not isinstance(response_data['documents'], list):
            print("ERROR: 'documents' field is not an array")
            return False
            
        print(f"✓ Found 'documents' array with {len(response_data['documents'])} items")
        
        # Check that the arrays have the same length
        if len(response_data['ids']) != len(response_data['documents']):
            print("ERROR: 'ids' and 'documents' arrays have different lengths")
            return False
            
        print("✓ 'ids' and 'documents' arrays have matching lengths")
        
        # Check key fields in document objects
        if len(response_data['documents']) > 0:
            doc = response_data['documents'][0]
            print(f"\nChecking document metadata structure...")
            
            required_fields = [
                'created_by', 'creationdate', 'creator', 'embedding_config',
                'file_id', 'hash', 'moddate', 'name', 'page', 'page_label',
                'producer', 'source', 'start_index', 'total_pages', 'trapped'
            ]
            
            missing_fields = []
            for field in required_fields:
                if field not in doc:
                    missing_fields.append(field)
                    
            if missing_fields:
                print(f"WARNING: Missing fields in document metadata: {missing_fields}")
            else:
                print("✓ All required document metadata fields are present")
                
            # Check embedding_config structure
            if 'embedding_config' in doc:
                embedding_config = doc['embedding_config']
                if isinstance(embedding_config, dict):
                    if 'engine' in embedding_config and 'model' in embedding_config:
                        print("✓ embedding_config has required fields")
                    else:
                        print("WARNING: embedding_config missing 'engine' or 'model' fields")
                else:
                    print("WARNING: embedding_config is not a dictionary")
        
        print("\n" + "="*50)
        print("SUCCESS: Response format is correct!")
        print("="*50)
        return True
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server. Please ensure the web API server is running.")
        print("Start the server with: python web_api.py")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Testing /invoke endpoint with return_original=True")
    print("="*50)
    
    success = test_invoke_endpoint()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        exit(1)