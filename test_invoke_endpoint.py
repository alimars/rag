#!/usr/bin/env python3
"""
Test script to verify the /invoke endpoint outputs the specified JSON format
when return_original=True
"""

import requests
import json
import uuid
import time
import subprocess
import sys
import os
from typing import Dict, Any


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if a string is a valid UUID"""
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def validate_response_format(response_data: Dict[str, Any]) -> tuple[bool, list]:
    """
    Validate that the response matches the required JSON format
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(response_data, dict):
        errors.append("Response is not a JSON object")
        return False, errors
    
    # Check required keys
    required_keys = ["ids", "documents"]
    for key in required_keys:
        if key not in response_data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors
    
    # Check ids array
    ids = response_data["ids"]
    if not isinstance(ids, list):
        errors.append("'ids' is not an array")
    else:
        # Check each ID is a valid UUID
        for i, doc_id in enumerate(ids):
            if not is_valid_uuid(doc_id):
                errors.append(f"ID at index {i} is not a valid UUID: {doc_id}")
    
    # Check documents array
    documents = response_data["documents"]
    if not isinstance(documents, list):
        errors.append("'documents' is not an array")
    else:
        # Check each document has required fields
        required_document_fields = [
            "created_by", "creationdate", "creator", "embedding_config",
            "file_id", "hash", "moddate", "name", "page", "page_label",
            "producer", "source", "start_index", "total_pages", "trapped"
        ]
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                errors.append(f"Document at index {i} is not a JSON object")
                continue
                
            # Check all required fields exist
            for field in required_document_fields:
                if field not in doc:
                    errors.append(f"Document at index {i} missing required field: {field}")
            
            # Validate specific fields
            if "created_by" in doc and not is_valid_uuid(doc["created_by"]):
                errors.append(f"Document at index {i} has invalid 'created_by' UUID: {doc['created_by']}")
                
            if "file_id" in doc and not is_valid_uuid(doc["file_id"]):
                errors.append(f"Document at index {i} has invalid 'file_id' UUID: {doc['file_id']}")
                
            # Check embedding_config structure
            if "embedding_config" in doc:
                embedding_config = doc["embedding_config"]
                if not isinstance(embedding_config, dict):
                    errors.append(f"Document at index {i} has invalid 'embedding_config' (not an object)")
                else:
                    if "engine" not in embedding_config:
                        errors.append(f"Document at index {i} missing 'embedding_config.engine'")
                    if "model" not in embedding_config:
                        errors.append(f"Document at index {i} missing 'embedding_config.model'")
            
            # Check numeric fields
            numeric_fields = ["page", "start_index", "total_pages"]
            for field in numeric_fields:
                if field in doc and not isinstance(doc[field], int):
                    errors.append(f"Document at index {i} has non-integer '{field}': {doc[field]}")
            
            # Check boolean fields
            boolean_fields = ["trapped"]
            for field in boolean_fields:
                if field in doc and not isinstance(doc[field], bool):
                    errors.append(f"Document at index {i} has non-boolean '{field}': {doc[field]}")
    
    return len(errors) == 0, errors


def start_server() -> subprocess.Popen:
    """Start the web API server"""
    print("Starting web API server...")
    
    # Change to the correct directory
    server_dir = os.path.join("advanced-rag-offline")
    
    # Start the server as a subprocess
    process = subprocess.Popen(
        [sys.executable, "web_api.py"],
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server time to start
    time.sleep(5)
    
    return process


def test_invoke_endpoint():
    """Test the /invoke endpoint with return_original=True"""
    server_process = None
    
    try:
        # Check if server is already running
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            if health_response.status_code == 200:
                print("Server is already running")
            else:
                print("Starting server...")
                server_process = start_server()
        except requests.exceptions.ConnectionError:
            print("Starting server...")
            server_process = start_server()
        
        # Test the /invoke endpoint
        print("Testing /invoke endpoint with return_original=True...")
        
        test_payload = {
            "query": "chemical weapons",
            "return_original": True
        }
        
        response = requests.post(
            "http://localhost:8000/invoke",
            json=test_payload,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Parse the response
        try:
            # The response is a JSON string, so we need to parse it
            response_data = response.json()
            print("✅ Received response from /invoke endpoint")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Response content: {response.text}")
            return False
        
        # Validate the response format
        is_valid, errors = validate_response_format(response_data)
        
        if is_valid:
            print("✅ Response format is valid")
            print(f"Found {len(response_data.get('ids', []))} document IDs")
            print(f"Found {len(response_data.get('documents', []))} documents")
            
            # Print sample of the response structure
            if response_data.get("documents"):
                print("\nSample document structure:")
                sample_doc = response_data["documents"][0]
                print(json.dumps(sample_doc, indent=2, ensure_ascii=False))
            
            return True
        else:
            print("❌ Response format validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Clean up server process if we started it
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


def main():
    """Main function to run the test"""
    print("Testing /invoke endpoint implementation")
    print("=" * 50)
    
    success = test_invoke_endpoint()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The implementation works as expected.")
    else:
        print("❌ Tests failed! Please check the implementation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())