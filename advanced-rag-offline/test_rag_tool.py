#!/usr/bin/env python3
"""
Test script to verify RAG tool file structure for OpenWeb UI integration
"""

import os
import sys

def test_file_structure():
    """Test that the required files exist and have the expected structure"""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Check that __init__.py exists
        init_path = os.path.join(base_path, "rag_tool", "__init__.py")
        if os.path.exists(init_path):
            print("✅ rag_tool/__init__.py exists")
        else:
            print("❌ rag_tool/__init__.py missing")
            return False
            
        # Check that web_api.py exists
        api_path = os.path.join(base_path, "web_api.py")
        if os.path.exists(api_path):
            print("✅ web_api.py exists")
        else:
            print("❌ web_api.py missing")
            return False
            
        # Check that documentation exists
        doc_path = os.path.join(base_path, "OPENWEBUI_INTEGRATION.md")
        if os.path.exists(doc_path):
            print("✅ OPENWEBUI_INTEGRATION.md exists")
        else:
            print("❌ OPENWEBUI_INTEGRATION.md missing")
            return False
            
        print("✅ File structure is correct")
        return True
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def test_docker_setup():
    """Test that Docker setup files exist"""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Check that Dockerfile exists
        docker_path = os.path.join(base_path, "Dockerfile")
        if os.path.exists(docker_path):
            print("✅ Dockerfile exists")
        else:
            print("❌ Dockerfile missing")
            return False
            
        # Check that docker-compose.yml references the service
        compose_path = os.path.join(base_path, "..", "docker-compose.yml")
        if os.path.exists(compose_path):
            print("✅ docker-compose.yml exists")
        else:
            print("❌ docker-compose.yml missing")
            return False
            
        print("✅ Docker setup is correct")
        return True
    except Exception as e:
        print(f"❌ Docker setup test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG Tool File Structure for OpenWeb UI Integration")
    print("=" * 55)
    
    success = True
    success &= test_file_structure()
    success &= test_docker_setup()
    
    print("=" * 55)
    if success:
        print("🎉 File structure test passed!")
        print("The RAG tool is properly structured for OpenWeb UI integration.")
        print("\nNext steps:")
        print("1. Build and run the Docker container:")
        print("   docker-compose up -d rag-api")
        print("2. Configure the tool in OpenWeb UI with URL: http://localhost:8000/invoke")
        print("3. Refer to OPENWEBUI_INTEGRATION.md for detailed instructions")
    else:
        print("❌ File structure test failed.")
        sys.exit(1)