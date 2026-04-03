#!/bin/bash

# PicoClaw Ollama Cloud Testing Script

set -e

echo "=============================================="
echo "PicoClaw Ollama Cloud Integration Test"
echo "=============================================="
echo ""

# Test 1: Build
echo "[TEST 1] Building PicoClaw..."
cd /Users/randy-mac/picoclaw-ollama-cloud
if make build > /dev/null 2>&1; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

# Test 2: API Connectivity
echo ""
echo "[TEST 2] Testing Ollama Cloud API connectivity..."
OLLAMA_API_KEY="ff6f7475fd2140a39ae462e57c10f2b2.fyzu2YtMwQ65Tp8FR3dFs28c"

response=$(curl -s -w "%{http_code}" https://ollama.com/v1/models \
    -H "Authorization: Bearer $OLLAMA_API_KEY" \
    -H "Content-Type: application/json" 2>/dev/null || echo "000")

http_code=$(echo "$response" | tail -c 4)

if [ "$http_code" = "200" ]; then
    echo "✅ API connectivity successful (HTTP $http_code)"
else
    echo "❌ API connectivity failed (HTTP $http_code)"
    echo "Response: $response"
fi

# Test 3: Simple Chat Completion
echo ""
echo "[TEST 3] Testing simple chat completion..."
chat_response=$(curl -s https://ollama.com/v1/chat/completions \
    -H "Authorization: Bearer $OLLAMA_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50
    }' 2>/dev/null)

if echo "$chat_response" | grep -q "choices"; then
    echo "✅ Chat completion successful"
    echo "Response preview: $(echo "$chat_response" | grep -o '"content":"[^"]*"' | head -1)"
else
    echo "❌ Chat completion failed"
    echo "Response: $chat_response"
fi

# Test 4: Tool Calling
echo ""
echo "[TEST 4] Testing tool calling..."
tool_response=$(curl -s https://ollama.com/v1/chat/completions \
    -H "Authorization: Bearer $OLLAMA_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        }]
    }' 2>/dev/null)

if echo "$tool_response" | grep -q "tool_calls\|finish_reason"; then
    echo "✅ Tool calling response received"
else
    echo "⚠️  Tool calling may not be supported or model returned direct response"
    echo "Response preview: $(echo "$tool_response" | head -c 200)"
fi

# Test 5: Model Availability
echo ""
echo "[TEST 5] Testing model availability..."
echo "Testing models:"
echo "  - gpt-oss:120b-cloud"
echo "  - kimi-k2.5:cloud"
echo "  - minimax-m2.7:cloud"

for model in "gpt-oss:120b-cloud" "kimi-k2.5:cloud" "minimax-m2.7:cloud"; do
    model_response=$(curl -s https://ollama.com/v1/chat/completions \
        -H "Authorization: Bearer $OLLAMA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 10}" 2>/dev/null)
    
    if echo "$model_response" | grep -q "error"; then
        echo "  ❌ $model - Not available or error"
    elif echo "$model_response" | grep -q "choices"; then
        echo "  ✅ $model - Available"
    else
        echo "  ⚠️  $model - Unknown status"
    fi
done

echo ""
echo "=============================================="
echo "Testing complete!"
echo "=============================================="
