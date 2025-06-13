"""
🔑 OpenAI API Key Configuration for Revolutionary Training
========================================================

This script helps you configure your OpenAI API key for the Revolutionary
LLM-GNN training system where GPT-4o actively controls the neural network.
"""

import os
import sys
from pathlib import Path


def setup_openai_key():
    """Setup OpenAI API key for Revolutionary training"""
    print("🔑 OpenAI API Key Configuration")
    print("=" * 50)
    
    # Check if key is already set
    current_key = os.getenv('OPENAI_API_KEY')
    if current_key:
        masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
        print(f"✅ OpenAI API key already set: {masked_key}")
        
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("✅ Using existing key")
            return current_key
    
    print("\n🎯 Enter your OpenAI API key:")
    print("   • Get it from: https://platform.openai.com/api-keys")
    print("   • Format: sk-...")
    print("   • Required for GPT-4o revolutionary control")
    
    api_key = input("\nOpenAI API Key: ").strip()
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid key format. Should start with 'sk-'")
        return None
    
    # Set environment variable for current session
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Create .env file for persistence
    env_file = Path('.env')
    env_content = []
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.readlines()
    
    # Remove existing OPENAI_API_KEY lines
    env_content = [line for line in env_content if not line.startswith('OPENAI_API_KEY=')]
    
    # Add new key
    env_content.append(f'OPENAI_API_KEY={api_key}\n')
    
    with open(env_file, 'w') as f:
        f.writelines(env_content)
    
    print(f"✅ OpenAI API key configured!")
    print(f"   • Environment variable: Set for current session")
    print(f"   • .env file: Created/updated for persistence")
    print(f"   • Masked key: {api_key[:8]}...{api_key[-4:]}")
    
    return api_key


def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🧪 Testing OpenAI API connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Test with a simple embedding request
        response = client.embeddings.create(
            input="Test connection for Revolutionary AI training",
            model="text-embedding-3-large"
        )
        
        print("✅ OpenAI API connection successful!")
        print(f"   • Model: text-embedding-3-large")
        print(f"   • Embedding dimension: {len(response.data[0].embedding)}")
        print("   • Ready for Revolutionary training!")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        print("   • Check your API key")
        print("   • Ensure you have credits")
        print("   • Verify internet connection")
        return False


def show_usage_guide():
    """Show how to use the Revolutionary training system"""
    print("\n🚀 REVOLUTIONARY TRAINING USAGE GUIDE")
    print("=" * 50)
    
    print("1️⃣ ENVIRONMENT SETUP:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   # OR use .env file (already created)")
    
    print("\n2️⃣ TRAINING COMMAND:")
    print("   cd testing")
    print("   python train_4090_ultra_scaled.py")
    
    print("\n3️⃣ REVOLUTIONARY FEATURES:")
    print("   🧠 GPT-4o actively controls GNN computation")
    print("   🔄 6-step iterative reasoning loops")
    print("   🎯 Real TSPLIB data (102 problems)")
    print("   🚀 Hierarchical memory + pointer networks")
    print("   ⚡ Neural search optimization")
    
    print("\n4️⃣ MONITORING:")
    print("   tensorboard --logdir=runs")
    print("   # Watch revolutionary LLM-GNN communication")
    
    print("\n5️⃣ COST ESTIMATION:")
    print("   • ~$0.10 per training batch (GPT-4o embeddings)")
    print("   • ~$5-20 for full training run")
    print("   • Revolutionary breakthrough: PRICELESS! 🎉")


if __name__ == "__main__":
    print("🚀 REVOLUTIONARY AI TRAINING SETUP")
    print("=" * 50)
    
    # Setup API key
    api_key = setup_openai_key()
    
    if api_key:
        # Test connection
        if test_openai_connection():
            show_usage_guide()
        else:
            print("\n❌ Setup incomplete - fix API connection first")
    else:
        print("\n❌ Setup failed - invalid API key")
    
    print("\n🎯 Ready for BREAKTHROUGH training!") 